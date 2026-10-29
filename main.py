import os
import random
import argparse
import yaml
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms

from datasets.imagenet import ImageNet
from datasets import build_dataset
from datasets.utils import build_data_loader
import clip
from utils import *
from models import *

logger = setup_logger("TIMO")

def get_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('--shot', dest='shot', type=int, default=1, help='shots number')
    parser.add_argument('--seed', dest='seed', type=int, default=1, help='seed')
    parser.add_argument('--dbg', dest='dbg', type=float, default=0, help='debug mode')
    parser.add_argument('--config', dest='config', help='settings of Tip-Adapter in yaml format')
    parser.add_argument('--timo_mode', type=str, choices=['tgi', 'igt', 'both', 'base'], default='base', help='TIMO mode: TGI-only, IGT-only, or both')
    parser.add_argument('--junk', type=int, default=0, help='Number of cross-class junk prompts to inject per class')

    args = parser.parse_args()
    return args


def main():

    # Load config file
    args = get_arguments()
    assert (os.path.exists(args.config))
    
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    cfg['shots'] = args.shot
    cfg['seed'] = args.seed
    cfg['dbg'] = args.dbg
    junk_prompts = args.junk

    if args.timo_mode == None:
        timo_mode = 'base'

    timo_mode = args.timo_mode
    logger.info("shots", cfg['shots'])
    logger.info("seed", cfg['seed'])
    logger.info("dbg", cfg['dbg'])
    logger.info("timo_mode", timo_mode)
    
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    cache_dir = os.path.join(f'./caches/{cfg["backbone"]}/{cfg["seed"]}/{cfg["dataset"]}')
    os.makedirs(cache_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir
    logger.info(cfg)

    # CLIP
    clip_model, preprocess = clip.load(cfg['backbone'])
    clip_model.eval()

    # Prepare dataset
    random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])

    # Textual features
    logger.info("\nGetting textual features as CLIP's classifier.")

    clip_weights_cupl_all = torch.load(cfg['cache_dir'] + "/text_weights_cupl_t_all.pt", weights_only=False)

    
    # --- EXTRA EXPERIMENT: CROSS-CLASS JUNK INJECTION ---
    if junk_prompts > 0:
        logger.info(f'Injecting {junk_prompts} cross-class junk prompts per class...')
        clip_weights_cupl_all = inject_junk(clip_weights_cupl_all, junk_prompts, cfg['seed'])

    
    classes, prompt_cupl_num, dim = clip_weights_cupl_all.shape

    logger.info(f"Textual features shape: {clip_weights_cupl_all.shape}. Number of classes in the dataset: {classes}. Number of prompts: {prompt_cupl_num}. Dimension of features: {dim}.")

    
    clip_weights_cupl = clip_weights_cupl_all.mean(dim=1).t() # average over prompts
    clip_weights_cupl = clip_weights_cupl / clip_weights_cupl.norm(dim=0, keepdim=True)
    
    # Construct the cache model by few-shot training set
    logger.info("\nConstructing cache model by few-shot visual features and labels.")
    cache_keys, cache_values = load_few_shot_feature(cfg)

    # Pre-load val features
    logger.info("\nLoading visual features and labels from val set.")
    val_features, val_labels = load_val_test_feature(cfg, "val")
    
    # Pre-load test features
    logger.info("\nLoading visual features and labels from test set.")
   
    test_features, test_labels = load_val_test_feature(cfg, "test")


    # ------------------------------------------ Fusion ------------------------------------------
    
    # Group images by class and compute the mean of their features to get image weights
    image_weights_all = torch.stack([cache_keys.t()[torch.argmax(cache_values, dim=1)==i] for i in range(classes)]) 


    # Build the average prototype for each class and normalize it
    image_weights = image_weights_all.mean(dim=1)
    image_weights = image_weights / image_weights.norm(dim=1, keepdim=True) 
    
    clip_weights_IGT, matching_score = image_guide_text(cfg,  clip_weights_cupl_all, image_weights, return_matching=True)
    clip_weights_IGT = clip_weights_IGT.t()

    metric = {}

    if junk_prompts > 0:
        _, igt_w = image_guide_text(cfg, clip_weights_cupl_all, image_weights, return_weights=True)
        metric['junk_weight_share'] = igt_w[:, -junk_prompts:].sum(dim=1).mean().item()
        metric['junk_equal_share'] = junk_prompts / prompt_cupl_num

    use_igt = timo_mode  == 'igt'
    use_tgi = timo_mode == 'tgi'
    use_base = timo_mode == 'base'
    use_both = timo_mode == 'both'


    if use_tgi:
         
        clip_text_weights = clip_weights_cupl
        keys, values = add_prompts_to_cache(cache_keys, cache_values, clip_weights_cupl_all, matching_score)

    elif use_igt:

        clip_text_weights = clip_weights_IGT
        keys, values = cache_keys, cache_values

    elif use_both:
        keys, values = add_prompts_to_cache(cache_keys, cache_values, clip_weights_cupl_all, matching_score)
        clip_text_weights = clip_weights_IGT

    elif use_base:
        keys, values = cache_keys, cache_values
        clip_text_weights = clip_weights_cupl


    acc_free = run_tip_adapter(cfg, keys, values, val_features, val_labels,  test_features, test_labels, clip_text_weights)
    metric['Tip_Adapter'] = acc_free

    if use_tgi or use_both:
        metric['GDA_CLIP'] = GDA_CLIP_TGI( cfg, val_features,val_labels,test_features,test_labels,clip_text_weights, clip_weights_cupl_all, matching_score,alpha_shift=False)
    else:
        acc_free = GDA_CLIP(cfg, val_features, val_labels, test_features, test_labels, clip_text_weights)
        metric['GDA_CLIP'] = acc_free
   
    
    # only run APE and TIMO if we are using the base run
    if use_base:
        # APE
        acc_free = APE(cfg, cache_keys, cache_values, val_features, val_labels,  test_features, test_labels, clip_text_weights)
        metric['APE'] = acc_free

        # ------------------------------------------ Ours ------------------------------------------
        # TIMO   

        acc_free = TIMO(cfg, val_features, val_labels, test_features, test_labels, clip_weights_IGT, clip_weights_cupl_all, matching_score, grid_search=False, is_print=True)
        metric['TIMO'] = acc_free


    if timo_mode == 'base':
        run_name = 'original'
    else:
        run_name = timo_mode

    if junk_prompts > 0:
        run_name = f'{run_name}_junk{junk_prompts}'

    save_log(cfg, metric, run_name=run_name)


    
if __name__ == '__main__':
    main()
