import torch
import torch.nn as nn
from torchprofile import profile_macs
from models.cifar10.vgg_cpd import vgg_16_bn_cpd
from models.cifar10.vgg import vgg_16_bn
import utils.common as utils
from ptflops import get_model_complexity_info


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
    original_model = vgg_16_bn().cuda()

    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(original_model, (3, 32, 32), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
    original_params = get_num_parameters(original_model)
    print('original_params = ', original_params)
    compress_rate = utils.get_cpr('[0.]*100]')
    for rank in range(1, 10):
        model = vgg_16_bn_cpd(rank, compress_rate).cuda()
        params = get_num_parameters(model)
        reduced = (1-params/original_params)*100
        # dummy_input = torch.randn(1, 3, 32, 32).cuda()
        # macs = get_model_macs(model, dummy_input)
        # print(params, macs)
        print(f'rank = {rank}, params = {params}, reduced = {reduced:.2f} %')
