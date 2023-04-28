import argparse
import torch
import torch.nn as nn
from torchprofile import profile_macs
from models.cifar10.vgg import vgg_16_bn
import utils.common as utils


def parse_args():
    parser = argparse.ArgumentParser('Compute model complexity')

    parser.add_argument('--arch', type=str, default='vgg_16_bn',
                        choices=('vgg_16_bn', 'resnet_56'), help='architecture')
    parser.add_argument('-r', '--rank', dest='rank', type=int, default=6,
                        help='use pre-specified rank for all layers')
    parser.add_argument('-cpr', '--compress_rate', type=str, default='[0.]*100',
                        help='list of compress rate of each layer')

    return parser.parse_args()


def get_num_parameters(model: nn.Module, count_nonzero_only=False) -> int:
    """
    calculate the total number of parameters of model
    :param count_nonzero_only: only count nonzero weights
    """
    num_counted_elements = 0
    for param in model.parameters():
        if count_nonzero_only:
            num_counted_elements += param.count_nonzero()
        else:
            num_counted_elements += param.numel()

    return num_counted_elements


def get_model_macs(model, inputs) -> int:

    return profile_macs(model, inputs)


if __name__ == '__main__':
    args = parse_args()

    original_model = vgg_16_bn().cuda()
    original_params = get_num_parameters(original_model)
    print('original_params = ', original_params)

    compress_rate = utils.get_cpr(args.compress_rate)

    model = vgg_16_bn(compress_rate, args.rank).cuda()
    print(model)
    params = get_num_parameters(model)
    reduced = (1-params/original_params)*100
    dummy_input = torch.randn(1, 3, 32, 32).cuda()
    # macs = get_model_macs(model, dummy_input)
    # model(dummy_input)
    print(f'params = {params}, reduced = {reduced:.2f} %')
