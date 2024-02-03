import tensorly as tl
from tensorly.decomposition import parafac
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from .CPDBlock import CPDBlock


tl.set_backend("pytorch")


def decompose(model: nn.Module, rank: int, n_iter_max=300, n_iter_singular_error=3):
    """
    Decompose a Conv2d model to a CPDBlock model.

    Args:
        model (nn.Module): a Conv2d form model.
        rank (int): rank.
        n_iter_max (int): max number of iterations for parafac.
        n_iter_singular_error (int): number of iterations for singular maxtrix error handler.

    Returns:
        model (nn.Module): a CPDBlock form model

    """

    for name, module in model._modules.items():
        if isinstance(module, nn.Conv2d):
            if (module.kernel_size[0] == 3):
                print(f'decomposing: {name}')
                model._modules[name] = conv_to_cpdblock(
                    module, rank, n_iter_max, n_iter_singular_error)
            else:
                print(
                    f'module {name} has kernel_size = {module.kernel_size}, passing')

        elif len(list(module.children())) > 0:
            print(f'recursing module: {name}')
            # recurse
            model._modules[name] = decompose(module, rank)

    return model


def conv_weights_to_factors(weights: torch.Tensor, rank: int, n_iter_max=300, n_iter_singular_error=3):
    """
    Decompose a Conv2d's weights to factors.

    Args:
        conv2d (nn.Conv2d): a Conv2d.
        rank (int): rank.
        n_iter_max (int): max number of iterations for parafac.
        n_iter_singular_error (int): number of iterations for singular maxtrix error handler.

    Returns:
        head_factor, body_factor, tail_factor: factors.

    """

    kernel_size = weights.size(2)
    in_channels = weights.size(1)
    out_channels = weights.size(0)
    device = None if weights.get_device() < 0 else weights.get_device()

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
                    head_factor[:, :, i], body_factor[:, :,
                                                      i], tail_factor[:, :, i] = factors
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

    return head_factor, body_factor, tail_factor


def conv_to_cpdblock(conv2d: nn.Conv2d, rank: int, n_iter_max=300, n_iter_singular_error=3):
    """
    Decompose a Conv2d with CPD.

    Args:
        conv2d (nn.Conv2d): a Conv2d.
        rank (int): rank.
        n_iter_max (int): max number of iterations for parafac.
        n_iter_singular_error (int): number of iterations for singular maxtrix error handler.

    Returns:
        cpdblock (CPDBlock): a CPDBlock.

    """

    kernel_size = conv2d.kernel_size[0]
    in_channels = conv2d.in_channels
    out_channels = conv2d.out_channels
    weights = conv2d.weight.data
    device = None if weights.get_device() < 0 else weights.get_device()

    head_factor, body_factor, tail_factor = conv_weights_to_factors(
        weights, rank, n_iter_max, n_iter_singular_error)

    biased = (conv2d.bias != None)
    # instantiate CPDBlock
    cpdblock = CPDBlock(in_channels, out_channels, rank, kernel_size,
                        conv2d.stride[0],
                        conv2d.padding[0],
                        biased,
                        device)

    # assign factors to CPDBlock's weights
    pointwise_weight, vertical_weight, horizontal_weight = factors_to_cpdblock_weights(
        head_factor, body_factor, tail_factor)
    cpdblock.feature.pointwise.weight.data = pointwise_weight
    cpdblock.feature.vertical.weight.data = vertical_weight
    cpdblock.feature.horizontal.weight.data = horizontal_weight

    if biased:
        cpdblock.bias.data = conv2d.bias.data

    return cpdblock


def cpdblock_weights_to_factors(pointwise_weight, vertical_weight, horizontal_weight, rank):
    """
    Reconstruct CPD factors from CPDBlock's weights.

    Args:
        pointwise_weight, vertical_weight, horizontal_weight: CPDBlock's weights.
        rank (int): rank.

    Returns:
        head_factor, body_factor, tail_factor: CPD factors.

    """

    in_channels = pointwise_weight.size(1)
    out_channels = int(pointwise_weight.size(0) / rank)
    kernel_size = vertical_weight.size(2)

    head_factor = pointwise_weight.squeeze(-1).squeeze(-1).reshape(
        rank, out_channels, in_channels).permute(2, 0, 1)
    body_factor = vertical_weight.squeeze(
        1).squeeze(-1).reshape(rank, out_channels, kernel_size).permute(2, 0, 1)
    tail_factor = horizontal_weight.squeeze(1).squeeze(
        1).reshape(rank, out_channels, kernel_size).permute(2, 0, 1)

    return head_factor, body_factor, tail_factor


def factors_to_cpdblock_weights(head_factor, body_factor, tail_factor):
    """
    Reconstruct CPDBlock's weights from CPD factors.

    Args:
        head_factor, body_factor, tail_factor: CPD factors.

    Returns:
        pointwise_weight, vertical_weight, horizontal_weight: CPDBlock's weights.

    """

    pointwise_weight = transform(head_factor).unsqueeze(-1).unsqueeze(-1)
    vertical_weight = transform(body_factor).unsqueeze(1).unsqueeze(-1)
    horizontal_weight = transform(tail_factor).unsqueeze(1).unsqueeze(2)

    return pointwise_weight, vertical_weight, horizontal_weight


def transform(factor):
    """
    Reshape and permute a factor of CPD

    Args:
        factor: a CPD factor of shape (x, rank, out_channels).
                x can be in_channels or kernel_size.

    Returns:
        output: a transformed tensor of shape (rank*out_channels, x).

    """

    rank = factor.size(1)
    out_channels = factor.size(2)
    x = factor.size(0)
    output = factor.reshape(x, rank*out_channels).permute(1, 0)

    return output
