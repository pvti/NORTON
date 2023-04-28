import tensorly as tl
from tensorly.decomposition import parafac
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from .CPDLayers import CPDLayer


tl.set_backend("pytorch")


def decompose(model, rank, n_iter_max=300, n_iter_singular_error=3):
    for name, module in model._modules.items():
        if isinstance(module, nn.Conv2d):
            model._modules[name] = decompose_conv_layer(
                module, rank, n_iter_max, n_iter_singular_error)
        elif len(list(module.children())) > 0:
            # recurse
            model._modules[name] = decompose(module, rank)

    return model


def decompose_conv_layer(layer: nn.Conv2d, rank: int, n_iter_max=300, n_iter_singular_error=3):
    """ Gets a Conv2D layer and a target rank, 
        returns a CPDLayer object with the decomposition
    """

    kernel_size = layer.kernel_size[0]
    Cin = layer.in_channels
    Cout = layer.out_channels
    W = layer.weight.data
    device = W.get_device()

    # Initialize the factor matrices with zeros
    head_factors = torch.zeros(
        (Cout, Cin, rank), device=device, requires_grad=False)
    body_factors = torch.zeros(
        (Cout, kernel_size, rank), device=device, requires_grad=False)
    tail_factors = torch.zeros(
        (Cout, kernel_size, rank), device=device, requires_grad=False)

    for i in tqdm(range(Cout)):
        weight = W[i, :, :, :]
        if torch.any(weight) != 0:
            success = False
            count = 0
            while not success and count < n_iter_singular_error:
                try:
                    _, factors = parafac(
                        weight, rank=rank, n_iter_max=n_iter_max, init='random')
                    head_factors[i], tail_factors[i], body_factors[i] = factors
                    success = True
                except torch._C._LinAlgError:
                    count += 1
                    pass

    assert not torch.isnan(head_factors).any(
    ), "head_factors tensor from parafac is nan"
    assert not torch.isnan(body_factors).any(
    ), "body_factors tensor from parafac is nan"
    assert not torch.isnan(tail_factors).any(
    ), "tail_factors tensor from parafac is nan"

    head_factors = head_factors.permute(1, 0, 2)
    body_factors = body_factors.permute(0, 2, 1)
    tail_factors = tail_factors.permute(0, 2, 1)

    biased = (layer.bias != None)
    cpd_layer = CPDLayer(Cin, Cout, rank, kernel_size,
                         layer.stride[0],
                         layer.padding[0],
                         head_factors.detach(),
                         body_factors.detach(),
                         tail_factors.detach(),
                         biased, layer.bias,
                         device)

    return cpd_layer
