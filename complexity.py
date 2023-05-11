import argparse
import torch
from models.cifar10 import *
from utils.common import get_cpr
from ptflops import get_model_complexity_info


def parse_args():
    parser = argparse.ArgumentParser('Compute model complexity')

    parser.add_argument('--arch', type=str, default='vgg_16_bn',
                        choices=('vgg_16_bn', 'resnet_56', 'densenet_40'), help='architecture')
    parser.add_argument('-r', '--rank', dest='rank', type=int, default=6,
                        help='use pre-specified rank for all layers')
    parser.add_argument('-cpr', '--compress_rate', type=str, default='[0.]*100',
                        help='list of compress rate of each layer')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    with torch.cuda.device(0):
        compress_rate = get_cpr(args.compress_rate)
        model = eval(args.arch)(compress_rate, args.rank)

        macs, params = get_model_complexity_info(model,
                                                 (3, 32, 32),
                                                 as_strings=True,
                                                 print_per_layer_stat=True,
                                                 verbose=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))
