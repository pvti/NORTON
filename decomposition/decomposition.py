import tensorly as tl
from tensorly.decomposition import parafac
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from .CPDLayers import CPDLayer


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
        returns a CPDLayer object with the decomposition """

    padding = layer.padding[0]
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
        # head_factors[i], tail_factors[i], body_factors[i] = parafac(
        #     W[i, :, :, :], rank=rank, init='random')[1]
        weight = W[i, :, :, :].to('cpu')
        factors = parafac(weight, rank=rank, n_iter_max=1000,
                          tol=1e-32, init='svd', svd='truncated_svd')[1]
        head_factors[i] = factors[0].clone().detach().to(device)
        body_factors[i] = factors[2].clone().detach().to(device)
        tail_factors[i] = factors[1].clone().detach().to(device)

    assert not torch.isnan(head_factors).any(
    ), "head_factors tensor from parafac is nan"
    assert not torch.isnan(body_factors).any(
    ), "body_factors tensor from parafac is nan"
    assert not torch.isnan(tail_factors).any(
    ), "tail_factors tensor from parafac is nan"

    head_factors = head_factors.permute(1, 0, 2)
    body_factors = body_factors.permute(0, 2, 1)
    tail_factors = tail_factors.permute(0, 2, 1)

    cpd_layer = CPDLayer(Cin, Cout, rank, kernel_size, padding, device)
    cpd_layer.head.weight.data.copy_(head_factors.detach())
    cpd_layer.body.weight.data.copy_(body_factors.detach())
    cpd_layer.tail.weight.data.copy_(tail_factors.detach())
    cpd_layer.tail.bias.data.copy_(layer.bias.data)

    return cpd_layer
