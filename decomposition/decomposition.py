import tensorly as tl
from tensorly.decomposition import parafac
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from .CPDLayers import CPDLayer


tl.set_backend("pytorch")


def decompose(model, rank):
    for name, module in model.features._modules.items():
        if isinstance(module, nn.Conv2d):
            cpd_layer = decompose_conv_layer(module, rank)
            model.features._modules[name] = cpd_layer
    return model


def reconstruct(model):
    for name, module in model.features._modules.items():
        if isinstance(module, CPDLayer):
            conv_layer = construct_conv_layer(module)
            model.features._modules[name] = conv_layer
    return model


def decompose_conv_layer(layer, rank):
    """ Gets a Conv2D layer and a target rank, 
        returns a CPDLayer object with the decomposition
    """

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
        weight = W[i, :, :, :].to('cpu')
        if torch.any(weight != 0):
            _, factors = parafac(weight, rank=rank)
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


def construct_conv_layer(layer: CPDLayer):
    """ Gets a CPDLayer, returns a Conv2d layer via CPD reconstruction
    """

    padding = layer.padding
    kernel_size = layer.kernel_size
    Cin = layer.in_channels
    Cout = layer.out_channels
    conv2d = nn.Conv2d(Cin, Cout, kernel_size=kernel_size, padding=padding)
    with torch.no_grad():
        weight = torch.zeros_like(conv2d.weight.data)

        # Get the factor matrices
        head_factors = layer.head.weight.data
        body_factors = layer.body.weight.data
        tail_factors = layer.tail.weight.data

        head_factors = head_factors.permute(1, 0, 2)
        body_factors = body_factors.permute(0, 2, 1)
        tail_factors = tail_factors.permute(0, 2, 1)

        for i in tqdm(range(Cout)):
            factors = [head_factors[i], tail_factors[i], body_factors[i]]
            weight[i] = tl.cp_to_tensor((None, factors), mask=None)

        conv2d.weight.data.copy_(weight)
        conv2d.bias.data.copy_(layer.tail.bias.data)

    return conv2d
