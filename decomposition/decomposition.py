import tensorly as tl
from tensorly.decomposition import parafac
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from .CPDBlock import CPDBlock


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


def decompose_conv_layer(conv2d: nn.Conv2d, rank: int, n_iter_max=300, n_iter_singular_error=3):
    """
    Decompose a Conv2d with CPD.

    Args:
        conv2d (nn.Conv2d): a Conv2d.
        rank (int): rank.
        n_iter_max (int): max number of iterations for parafac.
        n_iter_singular_error (int): number of iterations for singular maxtrix error handler.

    Returns:
        cpd_block (CPDBlock): a CPDBlock.

    """

    kernel_size = conv2d.kernel_size[0]
    in_channels = conv2d.in_channels
    out_channels = conv2d.out_channels
    weights = conv2d.weight.data
    device = weights.get_device()

    # Initialize the factor matrices with zeros
    head_factor = torch.zeros((in_channels, rank, out_channels), device=device)
    body_factor = torch.zeros((kernel_size, rank, out_channels), device=device)
    tail_factor = torch.zeros((kernel_size, rank, out_channels), device=device)

    for i in tqdm(range(out_channels)):
        weight = weights[i, :, :, :]
        if torch.any(weight) != 0:
            success = False
            count = 0
            while not success and count < n_iter_singular_error:
                try:
                    _, factors = parafac(
                        weight, rank=rank, n_iter_max=n_iter_max, init='random')
                    head_factor[:, :, i], body_factor[:, :, i], tail_factor[:, :, i] = factors
                    success = True
                except torch._C._LinAlgError:
                    count += 1
                    pass

    assert not torch.isnan(head_factor).any(
    ), "head_factor tensor from parafac is nan"
    assert not torch.isnan(body_factor).any(
    ), "body_factor tensor from parafac is nan"
    assert not torch.isnan(tail_factor).any(
    ), "tail_factor tensor from parafac is nan"

    biased = (conv2d.bias != None)
    # instantiate CPDBlock
    cpd_block = CPDBlock(in_channels, out_channels, rank, kernel_size,
                         conv2d.stride[0],
                         conv2d.padding[0],
                         biased,
                         device)

    # assign factors to CPDBlock's weights
    temp = rank*out_channels
    head_factor = head_factor.reshape(in_channels, temp).permute(1, 0).unsqueeze(-1).unsqueeze(-1)
    cpd_block.feature.pointwise.weight.data = head_factor

    body_factor = body_factor.reshape(kernel_size, temp).permute(1, 0).unsqueeze(1).unsqueeze(-1)
    cpd_block.feature.vertical.weight.data = body_factor

    tail_factor = tail_factor.reshape(kernel_size, temp).permute(1, 0).unsqueeze(1).unsqueeze(2)
    cpd_block.feature.horizontal.weight.data = tail_factor

    if biased:
        cpd_block.bias.data = conv2d.bias.data

    return cpd_block
