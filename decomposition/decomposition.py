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
    Cin = conv2d.in_channels
    Cout = conv2d.out_channels
    W = conv2d.weight.data
    device = W.get_device()

    # Initialize the factor matrices with zeros
    head_factors = torch.zeros((Cout, Cin, rank), device=device)
    body_factors = torch.zeros((Cout, kernel_size, rank), device=device)
    tail_factors = torch.zeros((Cout, kernel_size, rank), device=device)

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

    biased = (conv2d.bias != None)
    # instantiate CPDBlock
    cpd_block = CPDBlock(Cin, Cout, rank, kernel_size,
                         conv2d.stride[0],
                         conv2d.padding[0],
                         biased,
                         device)

    # assign factors to CPDBlock's weights
    for i in range(rank):
        head_factor = head_factors[:, :, i].unsqueeze(-1).unsqueeze(-1)
        cpd_block.head[i].weight.data = head_factor

    body_factors = body_factors.permute(2, 0, 1).unsqueeze(-1)
    cpd_block.body.weight.data = body_factors

    tail_factors = tail_factors.permute(0, 2, 1).unsqueeze(2)
    cpd_block.tail.weight.data = tail_factors

    cpd_block.tail.bias.data = conv2d.bias.data

    return cpd_block
