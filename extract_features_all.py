import os
import random
import json
import argparse
import yaml
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms

from datasets import build_dataset
from datasets.imagenet import ImageNet
from datasets.utils import build_data_loader
import clip
from utils import *

logger = setup_logger("TIMO")

def extract_few_shot_feature(cfg, clip_model, train_loader_cache, norm=True):

    logger.info('Extracting few-shot visual features and labels for cache model.')

    cache_keys = []
    cache_values = []
    with torch.no_grad():
        # Data augmentation for the cache model
        for augment_idx in range(cfg['augment_epoch']):
            train_features = []
            logger.info('Augment Epoch: {:} / {:}'.format(augment_idx, cfg['augment_epoch']))
            for i, (images, target) in enumerate(tqdm(train_loader_cache)):
                images = images.to(device())
                image_features = clip_model.encode_image(images) #100, 3, 224, 224
                train_features.append(image_features)
                if augment_idx == 0:
                    target = target.to(device())
                    cache_values.append(target)
            cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))
        
    cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
    if norm:
        cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
    cache_keys = cache_keys.permute(1, 0)
    cache_values = F.one_hot(torch.cat(cache_values, dim=0)).float()
    
    if norm:
        logger.info('Saving few-shot visual features and labels for cache model.')
        torch.save(cache_keys, cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
        torch.save(cache_values, cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")
        logger.info(f'✅ Saved few-shot visual features and labels for cache model.')
    else:
        logger.info('Saving few-shot visual features and labels for cache model.')
        torch.save(cache_keys, cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots_unnormed.pt")
        torch.save(cache_values, cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots_unnormed.pt")
        logger.info(f'✅ Saved few-shot visual features and labels for cache model.')
    return


def extract_few_shot_feature_all(cfg, clip_model, train_loader_cache, norm=True):

    logger.info('Extracting few-shot visual features and labels for cache model.')

    with torch.no_grad():      
        # Ours
        vecs = []
        labels = []
        for i in range(cfg["augment_epoch"]):
            for image, target in tqdm(train_loader_cache):
                image, target = image.to(device()), target.to(device())
                image_features = clip_model.encode_image(image)
                if norm:
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                vecs.append(image_features)
                labels.append(target)
        vecs = torch.cat(vecs)
        labels = torch.cat(labels)
        
        if norm:
            logger.info('Saving few-shot visual features and labels for cache model.')
            torch.save(vecs, cfg['cache_dir'] + "/" + f"{k}_vecs_f.pt")
            torch.save(labels, cfg['cache_dir'] + "/" + f"{k}_labels_f.pt")
            logger.info(f'✅ Saved few-shot visual features and labels for cache model.')
        else:
            logger.info('Saving few-shot visual features and labels for cache model.')
            torch.save(vecs, cfg['cache_dir'] + "/" + f"{k}_vecs_f_unnormed.pt")
            torch.save(labels, cfg['cache_dir'] + "/" + f"{k}_labels_f_unnormed.pt")
            logger.info(f'✅ Saved few-shot visual features and labels for cache model.')


def extract_val_test_feature(cfg, split, clip_model, loader, norm=True):
    
    logger.info(f'Extracting {split} visual features and labels.')

    features, labels = [], []
    with torch.no_grad():
        for i, (images, target) in enumerate(tqdm(loader)):
            images, target = images.to(device()), target.to(device())
            image_features = clip_model.encode_image(images)
            if norm:
                image_features /= image_features.norm(dim=-1, keepdim=True)
            features.append(image_features)
            labels.append(target)
    features, labels = torch.cat(features), torch.cat(labels)
    logger.info(f'Saving {split} visual features and labels.')

    if norm:
        logger.info(f'Saving {split} visual features and labels.')
        torch.save(features, cfg['cache_dir'] + "/" + split + "_f.pt")
        torch.save(labels, cfg['cache_dir'] + "/" + split + "_l.pt")
        logger.info(f'✅ Saved {split} visual features and labels.')

    else:
        logger.info(f'Saving {split} visual features and labels.')

        torch.save(features, cfg['cache_dir'] + "/" + split + "_f_unnormed.pt")
        torch.save(labels, cfg['cache_dir'] + "/" + split + "_l_unnormed.pt")

        logger.info(f'✅ Saved {split} visual features and labels.')

    return


def extract_text_feature(cfg, classnames, prompt_paths, clip_model, template, use_gpt_prompt=True):

    logger.info('Extracting text features for cache model.')

    prompts = []
    for prompt_path in prompt_paths:
        f = open(prompt_path)
        prompts.append(json.load(f))
        
    with torch.no_grad():
        clip_weights = []
        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            
            template_texts = [t.format(classname) for t in template]
            if use_gpt_prompt:
                texts = template_texts
                for prompt in prompts:
                    texts += prompt[classname]                    
            else:
                texts = template_texts
        
            texts_token = clip.tokenize(texts, truncate=True).to(device())
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts_token)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).to(device())
    if use_gpt_prompt:
        logger.info('Saving text features for cache model with GPT prompts...')
        torch.save(clip_weights, cfg['cache_dir'] + "/text_weights_gpt_t.pt")
        logger.info('✅ Saved text features for cache model with GPT prompts.')
    else:
        logger.info('Saving text features for cache model.')
        torch.save(clip_weights, cfg['cache_dir'] + "/text_weights_t.pt")
        logger.info('✅ Saved text features for cache model.')
    return


def extract_text_feature_all(cfg, classnames, prompt_paths, clip_model, template, norm=True):

    logger.info('Extracting text features for cache model.')
    prompts = []
    for prompt_path in prompt_paths:
        f = open(prompt_path)
        prompts.append(json.load(f))
        
    with torch.no_grad():
        clip_weights = []
        min_len = 1000
        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            
            template_texts = [t.format(classname) for t in template]
            # cupl_texts = prompts[0][classname]
            # waffle_texts = prompts[1][classname]
            # texts = template_texts + cupl_texts + waffle_texts
            texts = template_texts
            for prompt in prompts:
                texts += prompt[classname] 
        
            texts_token = clip.tokenize(texts, truncate=True).to(device())
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts_token)
            if norm:
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            min_len = min(min_len, len(class_embeddings))
            clip_weights.append(class_embeddings)
        
        for i in range(len(clip_weights)):
            clip_weights[i] = clip_weights[i][:min_len]

        clip_weights = torch.stack(clip_weights, dim=0).to(device())
        print(clip_weights.shape)
        
    logger.info('✅ Extracted text features for cache model.')

    if norm:
        logger.info('Saving text features for cache model.')
        torch.save(clip_weights, cfg['cache_dir'] + "/text_weights_cupl_t_all.pt")
        logger.info('✅ Saved text features for cache model.')
    else:

        logger.info('Saving text features for cache model.')
        torch.save(clip_weights, cfg['cache_dir'] + "/text_weights_cupl_t_all_unnormed.pt")
        logger.info('✅ Saved text features for cache model.')
    return


if __name__ == '__main__':
    
    backbone = 'RN50'
    seeds = [1, 2, 3]
    datasets = ['caltech101', 'dtd', 'oxford_pets']
    k_shot = [1, 4, 16]
    norm = True
    data_path = '../DATA'

    clip_model, preprocess = clip.load(backbone, device=device()) # Load the clip model
    clip_model.eval() # switch to evaluation mode

    logger.info(f'Extracting features for backbone: {backbone}, seeds: {seeds}, datasets: {datasets}, k_shot: {k_shot}, norm: {norm}')

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),std=(0.26862954, 0.26130258, 0.27577711))
        ])

    for seed in seeds:
        for name in datasets:
            cfg = yaml.load(open(f'configs/{name}.yaml', 'r'), Loader=yaml.Loader)
            cache_dir = os.path.join(f'./caches/{backbone}/{seed}', name)
            os.makedirs(cache_dir, exist_ok=True)
            cfg['cache_dir'] = cache_dir
            cfg['backbone'] = backbone
            cfg['seed'] = seed
            cfg['dataset'] = name

            for k in k_shot:
                random.seed(seed)
                torch.manual_seed(seed)
                cfg['shots'] = k
                print(cfg)

                dataset = build_dataset(name, data_path, k)
                val_loader = build_data_loader(data_source=dataset.val, batch_size=64, is_train=False, tfm=preprocess, shuffle=False)
                test_loader = build_data_loader(data_source=dataset.test, batch_size=64, is_train=False, tfm=preprocess, shuffle=False)
                train_loader_cache = build_data_loader(data_source=dataset.train_x, batch_size=256, tfm=train_transform, is_train=True, shuffle=False)

                print('\nConstructing cache model by few-shot visual features and labels.')
                extract_few_shot_feature(cfg, clip_model, train_loader_cache)
                extract_few_shot_feature_all(cfg, clip_model, train_loader_cache, norm=norm)

                print('\nLoading visual features and labels from val and test set.')
                extract_val_test_feature(cfg, 'val', clip_model, val_loader, norm=norm)
                extract_val_test_feature(cfg, 'test', clip_model, test_loader, norm=norm)

                extract_text_feature(cfg, dataset.classnames, [dataset.cupl_path], clip_model, dataset.template, use_gpt_prompt=False)
                extract_text_feature(cfg, dataset.classnames, [dataset.cupl_path], clip_model, dataset.template)
                extract_text_feature_all(cfg, dataset.classnames, [dataset.cupl_path], clip_model, dataset.template, norm=norm)

    logger.info(f'✅ Finished extracting features for backbone: {backbone}, seeds: {seeds}, datasets: {datasets}, k_shot: {k_shot}, norm: {norm}')