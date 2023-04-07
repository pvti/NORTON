import torch
import torch.nn as nn
from torchprofile import profile_macs


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
