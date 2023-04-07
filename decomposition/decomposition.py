import tensorly as tl
from tensorly.decomposition import parafac
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from .CPDLayers import CPDHead, CPDBody, CPDTail


def cp_decompose_model(model, rank, exclude_first_conv=False, passed_first_conv=False):
    for name, module in model._modules.items():
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name] = cp_decompose_model(
                module, rank, exclude_first_conv, passed_first_conv)
        elif type(module) == nn.Conv2d:
            if passed_first_conv is False:
                passed_first_conv = True
                if exclude_first_conv is True:
                    continue

            conv_layer = module
            decomposed = cp_decomposition_conv_layer(conv_layer, rank)
            model._modules[name] = decomposed

    return model


def cp_decomposition_conv_layer(layer, rank):
    """ Gets a Conv2D layer and a target rank, 
        returns a nn.Sequential object with the decomposition """

    padding = layer.padding[0]
    kernel_size = layer.kernel_size[0]
    Cin = layer.in_channels
    Cout = layer.out_channels
    W = layer.weight.data
    device = W.get_device()

    # Initialize the factor matrices with zeros
    head_factors = torch.zeros(Cout, Cin, rank, device=device)
    body_factors = torch.zeros(Cout, kernel_size, rank, device=device)
    tail_factors = torch.zeros(Cout, kernel_size, rank, device=device)

    for i in tqdm(range(Cout)):
        head_factors[i], tail_factors[i], body_factors[i] = parafac(
            W[i, :, :, :], rank=rank, init='random')[1]

    assert not torch.isnan(head_factors).any(
    ), "head_factors tensor from parafac is nan"
    assert not torch.isnan(body_factors).any(
    ), "body_factors tensor from parafac is nan"
    assert not torch.isnan(tail_factors).any(
    ), "tail_factors tensor from parafac is nan"

    head_factors = head_factors.permute(1, 0, 2)
    body_factors = body_factors.permute(0, 2, 1)
    tail_factors = tail_factors.permute(0, 2, 1)

    head = CPDHead(Cin, Cout, rank, padding, head_factors)
    body = CPDBody(Cin, Cout, rank, kernel_size, padding, body_factors)
    tail = CPDTail(Cin, Cout, rank, kernel_size, padding,
                   tail_factors, layer.bias.data)

    new_layers = [head, body, tail]

    return nn.Sequential(*new_layers)
