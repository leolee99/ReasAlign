import torch
import argparse
import random
import numpy as np
import logging

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    return seed

def get_logger(filename=None):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(asctime)s - %(levelname)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
    if filename is not None:
        handler = logging.FileHandler(filename)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logging.getLogger().addHandler(handler)
    return logger

def get_args(description='DRIFT'):
    parser = argparse.ArgumentParser(description=description)
    # Eval Setting
    parser.add_argument('--dataset', type=str, default='AlpacaEval', help='select the dataset from [alpaca, sep_util, sep_sec, cyse]')
    parser.add_argument('--defense_model', type=str, default=None, help='the path to your defensive model')
    parser.add_argument('--do_judge', action='store_true', help='whether if use judge model')
    parser.add_argument('--judge_model', type=str, default=None, help='the path to your judge model')
    parser.add_argument('--node_scale', type=int, default=3, help='The default node scale for selecting')
    parser.add_argument('--eval_model', type=str, default="gpt-4o-mini", help='Which model do you want for evaluation.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for inference.')

    # Environment
    parser.add_argument('--seed', type=int, default=98, help='Random Seed.')

    args = parser.parse_args()

    return args