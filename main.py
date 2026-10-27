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
    parser.add_argument('--timo_mode', type=str, choices=['tgi', 'igt', 'both'], default='both', help='TIMO mode: TGI-only, IGT-only, or both')

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

    if args.timo_mode == None:
        timo_mode = 'both'

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
    
    # ------------------------------------------ Baseline ------------------------------------------
    # Tip-Adapter
    acc_free = run_tip_adapter(cfg, cache_keys, cache_values, val_features, val_labels,  test_features, test_labels, clip_weights_cupl)
    metric['Tip_Adapter'] = acc_free
    
    # APE
    acc_free = APE(cfg, cache_keys, cache_values, val_features, val_labels,  test_features, test_labels, clip_weights_cupl)
    metric['APE'] = acc_free
    
    # GDA-CLIP
    acc_free = GDA_CLIP(cfg, val_features, val_labels, test_features, test_labels, clip_weights_cupl)
    metric['GDA_CLIP'] = acc_free
    
    # ------------------------------------------ Ours ------------------------------------------
    # TIMO   

    # Use igt or both
    if timo_mode in {'igt', 'both'}:
        clip_weights = clip_weights_IGT
    else:
        # use only tgi
        clip_weights = clip_weights_cupl

    acc_free = TIMO(cfg, val_features, val_labels, test_features, test_labels,  clip_weights, clip_weights_cupl_all, matching_score, grid_search=False, is_print=True, timo_mode=timo_mode)
    metric['TIMO'] = acc_free

    # TIMO-S
    
    ''' 
    #TODO  SEARCH FOR GAMMA

    clip_weights_IGT, matching_score = image_guide_text_search(cfg, 
        clip_weights_cupl_all, val_features, val_labels, image_weights)
    acc_free = TIMO(cfg, val_features, val_labels, test_features, test_labels, 
        clip_weights_IGT, clip_weights_cupl_all, matching_score, 
        grid_search=True, n_quick_search=10, is_print=True)
    metric['TIMO_S'] = acc_free
    '''

    if timo_mode == 'both':
        run_name = 'original'
    else:
        run_name = timo_mode

    save_log(cfg, metric, run_name=run_name)


    
if __name__ == '__main__':
    main()