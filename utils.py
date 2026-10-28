import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import logging
from pathlib import Path
from datetime import datetime

STAMP = datetime.now().strftime('%Y-%m-%d-%HH%M')  # fixed once per run

def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc


def cal_criterion(cfg, clip_weights, cache_keys, only_use_txt=True, training_free=True, force=False, with_IE=False):
    
    feat_dim, classes = clip_weights.shape
    text_feat = clip_weights.t().unsqueeze(1)
    cache_feat = cache_keys.reshape(classes, -1, feat_dim)
    
    save_path = f'./caches/{cfg["backbone"]}/{cfg["seed"]}/{cfg["dataset"]}'
    if with_IE:
        save_file = '{}/criterion_{}_{}shot_IE.pt'.format(save_path, cfg['backbone'].replace('/', ''), cfg['shots'])
    else:
        save_file = '{}/criterion_{}_{}shot_Pure.pt'.format(save_path, cfg['backbone'].replace('/', ''), cfg['shots'])
        
    if os.path.exists(save_file) and not force:
        print('Loading criterion...')
        sim = torch.load(save_file, weights_only=False)
    elif only_use_txt:
        print('Calculating criterion...')
        
        feats = text_feat.squeeze()
        
        sim_sum = torch.zeros((feat_dim)).to(device())
        count = 0
        for i in range(classes):
            for j in range(classes):
                if i != j:
                    sim_sum += feats[i, :] * feats[j, :]
                    count += 1
        sim = sim_sum / count
        torch.save(sim, save_file)
    else:
        print('Calculating criterion...')
        
        feats = torch.cat([text_feat, cache_feat], dim=1)
        samp_num = feats.shape[1]
        
        sim_sum = torch.zeros((feat_dim)).to(device())
        count = 0
        for i in range(classes):
            for j in range(classes):
                for m in range(samp_num):
                    for n in range(samp_num):
                        if i != j:
                            sim_sum += feats[i, m, :] * feats[j, n, :]
                            count += 1
        sim = sim_sum / count
        torch.save(sim, save_file)

    criterion = (-1) * cfg['w'][0] * sim + cfg['w'][1] * torch.var(clip_weights, dim=1)
    
    ratio = 1024 / feat_dim
    if training_free:
        _, indices = torch.topk(criterion, k=int(cfg['training_free_feat_num'] // ratio))
    else: 
        _, indices = torch.topk(criterion, k=int(cfg['training_feat_num'] // ratio))
    return indices


def load_text_feature(cfg):
    save_path = cfg['cache_dir'] + "/text_weights_gpt_t.pt"
    clip_weights = torch.load(save_path, weights_only=False)
    return clip_weights


def load_few_shot_feature(cfg, norm=True):
    if norm:
        cache_keys = torch.load(cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt", weights_only=False)
        cache_values = torch.load(cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt", weights_only=False)
    else:
        cache_keys = torch.load(cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots_unnormed.pt", weights_only=False)
        cache_values = torch.load(cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots_unnormed.pt", weights_only=False)
    return cache_keys, cache_values


def load_val_test_feature(cfg, split, norm=True):
    if norm:
        features = torch.load(cfg['cache_dir'] + "/" + split + "_f.pt", weights_only=False)
        labels = torch.load(cfg['cache_dir'] + "/" + split + "_l.pt", weights_only=False)
    else:
        features = torch.load(cfg['cache_dir'] + "/" + split + "_f_unnormed.pt", weights_only=False)
        labels = torch.load(cfg['cache_dir'] + "/" + split + "_l_unnormed.pt", weights_only=False)
    return features, labels


def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)

def accuracy(shot_logits, cache_values, topk=(1,)):
    target = cache_values.topk(max(topk), 1, True, True)[1].squeeze()
    pred = shot_logits.topk(max(topk), 1, True, True)[1].squeeze()
    idx = (target != pred)
    return idx


def search_hp(cfg, cache_keys, cache_values, features, labels, clip_weights, adapter=None):

    if cfg['search_hp'] == True:
    
        beta_list = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1 for i in range(cfg['search_step'][0])]
        alpha_list = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1 for i in range(cfg['search_step'][1])]

        best_acc = 0
        best_beta, best_alpha = 0, 0

        for beta in beta_list:
            for alpha in alpha_list:
                if adapter:
                    affinity = adapter(features)
                else:
                    affinity = features @ cache_keys

                cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
                clip_logits = 100. * features @ clip_weights
                tip_logits = clip_logits + cache_logits * alpha
                acc = cls_acc(tip_logits, labels)
            
                if acc > best_acc:
                    print("New best setting, beta: {:.2f}, alpha: {:.2f}; accuracy: {:.2f}".format(beta, alpha, acc))
                    best_acc = acc
                    best_beta = beta
                    best_alpha = alpha

        print("\nAfter searching, the best accuracy: {:.2f}.\n".format(best_acc))

    return best_beta, best_alpha


# t_features [c,p,d]
# v_features [c,n,d] or [c,d]
def image_guide_text(cfg, t_features, v_features, gamma=-1,  return_weights=False, return_matching=False):
   
    t_features = t_features / t_features.norm(dim=-1, keepdim=True)
    
    if gamma == -1:
        if cfg['dataset'] == "oxford_flowers":
            gamma = 100
        else:
            gamma = 50
        
    classes, prompt_num, feat_dim = t_features.shape # c, p, d

    if len(v_features.shape) == 3:

        v_features = v_features.mean(dim=1) # c,d
        v_features = v_features / v_features.norm(dim=-1, keepdim=True)

    weights = torch.ones(classes, prompt_num).to(t_features.dtype).to(t_features.device) # c, p

    v_features = v_features.to(t_features.dtype)    
    t_features = t_features / t_features.norm(dim=-1, keepdim=True)
    
    matching_score = []

    for c in range(classes):
 
        weights[c] = (v_features[c:c+1] @ t_features[c].t()).squeeze(dim=0)
        matching_score.append(weights[c].clone())
        weights[c] = weights[c] / weights[c].norm() 
        weights[c] = F.softmax(weights[c] * gamma, dim=0)

    matching_score = torch.stack(matching_score, dim=0) # N, P
    
    for weights in [weights]:
        normed_weights = weights
        normed_clip_weights = torch.einsum("cp, cpd-> cd", normed_weights, t_features)
        normed_clip_weights = normed_clip_weights / normed_clip_weights.norm(dim=-1, keepdim=True)

    
    if return_matching:
        return normed_clip_weights, matching_score
    elif return_weights:
        return normed_clip_weights, normed_weights
    else:
        return normed_clip_weights


def build_tgi_features(image_features, text_features):
    """
    image_features: [classes, shots, dim]
    text_features:  [classes, prompts, dim]
    """
    # Class-level visual prototypes
    visual_prototypes = image_features.mean(dim=1)
    visual_prototypes = F.normalize(visual_prototypes, dim=-1)

    # Match each text prompt to its class's visual prototype
    similarities = torch.einsum(
        "cpd,cd->cp", text_features, visual_prototypes
    )

    # Weight text features using the matching scores
    weighted_text = text_features * similarities.unsqueeze(-1)

    # Append weighted text features to the image features
    refined_features = torch.cat(
        [image_features, weighted_text], dim=1
    )

    return refined_features, similarities


def add_prompts_to_cache(cache_keys, cache_values, prompts_all, matching_score):
    # cache_keys: D x (N*K), cache_values: (N*K) x N
    # prompts_all: N x P x D, matching_score: N x P
    classes, p, dim = prompts_all.shape
    weighted = (prompts_all * matching_score.unsqueeze(-1)).reshape(classes * p, dim)
    labels = torch.arange(classes).repeat_interleave(p)
    values = F.one_hot(labels, num_classes=classes).to(cache_values.dtype).to(cache_values.device)
    keys = torch.cat([cache_keys, weighted.t().to(cache_keys.dtype).to(cache_keys.device)], dim=1)
    return keys, torch.cat([cache_values, values], dim=0)


def image_guide_text_search(cfg, clip_weights_cupl_all, val_features, val_labels, image_weights):
    best_acc = 0
    best_gamma = 0
    for gamma in range(5, 101, 5):
        clip_weights_cupl_IGT, matching_score = image_guide_text(cfg, 
            clip_weights_cupl_all, image_weights, return_matching=True, gamma=gamma)
        clip_weights_cupl_IGT = clip_weights_cupl_IGT.t() # D, C
        
        val_logits = val_features @ clip_weights_cupl_IGT # N, C
        acc = (val_logits.argmax(-1) == val_labels).sum() / len(val_labels)
        
        if acc > best_acc:
            best_acc = acc
            best_gamma = gamma
    print("best_gamma:", best_gamma)
    clip_weights_cupl_IGT, matching_score = image_guide_text(cfg, 
        clip_weights_cupl_all, image_weights, return_matching=True, gamma=best_gamma)
    clip_weights_cupl_IGT = clip_weights_cupl_IGT.t()
    return clip_weights_cupl_IGT,matching_score


# anchor K,D
def vec_sort(vecs_t, matching_score):
    classes, prompt_num, dim = vecs_t.shape # N,P,D
    
    weights, sorted_idx = torch.topk(matching_score, k=prompt_num, dim=-1)
    sort_vecs_t = []
    for c in range(classes):
        sort_vecs_t.append(vecs_t[c][sorted_idx[c]].clone())
    sort_vecs_t = torch.stack(sort_vecs_t, dim=0)
    
    if len(sort_vecs_t.shape) == 2:
        sort_vecs_t = sort_vecs_t.unsqueeze(1)
        
    return sort_vecs_t, weights


def save_log(cfg, metric:dict, run_name:str="original"):
    
    out_dir = f'outputs/{run_name}'
    os.makedirs(out_dir, exist_ok=True)
    
    for key in metric.keys():
        file_path = f'{out_dir}/{key}.txt'
        with open(file_path, 'a') as f:
            f.write(f"{cfg['dataset']}_{cfg['shots']}_{cfg['seed']}: {metric[key]}\n")
            

def device():
    return "cuda" if torch.cuda.is_available() else "cpu"






def _log_with_args(log_method, *args, **kwargs):
    """Support print-style and standard logging-style arguments."""

    if not args:
        log_method(**kwargs)
        return

    message = str(args[0])

    if len(args) == 1:
        log_method(message, **kwargs)
        return

    try:
        # Standard logging: logger.info("Value: %s", value)
        formatted_message = message % args[1:]
    except (TypeError, ValueError, KeyError):
        # Print-style: logger.info("Value:", value)
        formatted_message = " ".join(str(arg) for arg in args)

    log_method(formatted_message, **kwargs)

def inject_junk(t_all, m, seed):
    # append m prompts borrowed from other classes to every class
    if m == 0:
        return t_all
    g = torch.Generator().manual_seed(seed)
    c, p, _ = t_all.shape
    extra = []
    for i in range(c):
        donor = (i + torch.randint(1, c, (1,), generator=g).item()) % c
        idx = torch.randint(0, p, (m,), generator=g)
        extra.append(t_all[donor, idx])
    return torch.cat([t_all, torch.stack(extra)], dim=1)


def setup_logger(name="TIMO", log_dir="../DATA/logs", level=logging.INFO):
    import os
    from datetime import datetime
    from functools import partial
    from types import MethodType

    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(level)
        logger.propagate = False

        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(log_dir, f"{name}_{timestamp}.log")

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        file_handler = logging.FileHandler(
            log_path, mode="a", encoding="utf-8"
        )
        file_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    # Keep the original logging methods, then wrap them.
    for method_name in ("debug", "info", "warning", "error", "critical", "exception"):
        original_method = getattr(logger, method_name)

        def make_wrapper(method):
            def wrapper(*args, **kwargs):
                _log_with_args(method, *args, **kwargs)
            return wrapper

        setattr(logger, method_name, make_wrapper(original_method))

    # Support the older .warn() alias as well.
    logger.warn = logger.warning

    return logger