# coding=utf-8

import argparse
import random

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser()

    # Data Splits
    parser.add_argument('--train', help='train model', action='store_true')
    parser.add_argument('--test', help='test model', action='store_true')
    parser.add_argument('--atypical_test', help='test model on atypical task', action='store_true')

    # Model architectures
    parser.add_argument('--svte', help='use Relative-Spatial Transformer Encoder', action='store_true')
    parser.add_argument('--vte', help='use Transformer Encoder', action='store_true')

    # Folder Paths
    parser.add_argument("--annotation_path", help='path to the folder which contains the atypicality annotation')
    parser.add_argument("--atypical_test_path", help='path to the file which contains the atypicality annotation')
    parser.add_argument("--img_folder",  help='path to the folder which contains Ads images')
    parser.add_argument("--feat_folder", help='path to the folder which contains Ads image features')
    parser.add_argument('--output_dir', type=str, help='path to the folder which saves output')

    # Training Hyper-parameters
    parser.add_argument('--cartesian', help='use cartesian coordinate', action='store_true')
    parser.add_argument('--polar', help='use polar coordinate', action='store_true')
    parser.add_argument('--iou', help='compute IoU as relative positional encoding', action='store_true')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--batch_size_eval', type=int, default=128)
    parser.add_argument('--n_layer', help='number of transformer encoder layers', type=int, default=1)
    parser.add_argument('--d_model', type=int, default=768)
    parser.add_argument('--d_feat', type=int, default=2048)
    parser.add_argument('--d_pos', type=int, default=4)
    parser.add_argument('--n_mask', type=int, default=5)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--seed', type=int, default=4545, help='random seed')
    parser.add_argument('--ngpu', type=int, default=1, help='Number of GPUs available. Use 0 for CPU mode.')

    # Model Loading
    parser.add_argument('--load_model', type=str, default=None,
                        help='Load the model weights.')

    # Training configuration
    parser.add_argument('--num_workers', default=2)

    # Parse the arguments.
    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    return args


args = parse_args()
