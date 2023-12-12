import tensorly as tl
from tensorly.decomposition import partial_tucker
import torch
import torch.nn as nn


tl.set_backend("pytorch")


def tucker_decompose_model(model: nn.Module, rank_divisor: float):
    """
    Decompose a Conv2d model to a Tucker model.

    Args:
        model (nn.Module): a Conv2d form model.
        rank_divisor (float): rank divisor.

    Returns:
        model (nn.Module): a Tucker form model

    """
    for name, module in model._modules.items():
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name] = tucker_decompose_model(module, rank_divisor)
        elif type(module) == nn.Conv2d:
            conv_layer = module
            decomposed = tucker_decomposition_conv_layer(conv_layer, rank_divisor)
            model._modules[name] = decomposed

    return model


def tucker_decomposition_conv_layer(layer, rank_divisor):
    """
    Gets a conv layer, returns a nn.Sequential object with the Tucker decomposition.
    """
    C_out = layer.weight.data.size(0)
    C_in = layer.weight.data.size(1)
    rank = [int(C_out / rank_divisor), int(C_in / rank_divisor)]
    (core, [last, first]), _ = partial_tucker(
        layer.weight.data, modes=[0, 1], rank=rank, init="svd"
    )

    # A pointwise convolution that reduces the channels from S to R3
    first_layer = torch.nn.Conv2d(
        in_channels=first.shape[0],
        out_channels=first.shape[1],
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=layer.dilation,
        bias=False,
    )

    # A regular 2D convolution layer with R3 input channels
    # and R3 output channels
    core_layer = torch.nn.Conv2d(
        in_channels=core.shape[1],
        out_channels=core.shape[0],
        kernel_size=layer.kernel_size,
        stride=layer.stride,
        padding=layer.padding,
        dilation=layer.dilation,
        bias=False,
    )

    # A pointwise convolution that increases the channels from R4 to T
    last_layer = torch.nn.Conv2d(
        in_channels=last.shape[1],
        out_channels=last.shape[0],
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=layer.dilation,
        bias=True,
    )

    if layer.bias is not None:
        last_layer.bias.data = layer.bias.data

    first_layer.weight.data = torch.transpose(first, 1, 0).unsqueeze(-1).unsqueeze(-1)
    last_layer.weight.data = last.unsqueeze(-1).unsqueeze(-1)
    core_layer.weight.data = core

    new_layers = [first_layer, core_layer, last_layer]

    return nn.Sequential(*new_layers)
