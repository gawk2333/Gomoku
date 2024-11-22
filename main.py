import argparse
import os
import numpy as np
import game_env
import copy
from random import seed
import torch

parser = argparse.ArgumentParser(description='param')
parser.add_argument("--train", type=int, default=1)  # train=1: training phase
args = parser.parse_args()
print(args)


def evaluate(args):
    return 0

if __name__ == '__main__':
    g_env = game_env.Environment()
    # seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)

    if args.train == 1:
        g_env.train(args)

    else:
        g_env.evaluate(args)