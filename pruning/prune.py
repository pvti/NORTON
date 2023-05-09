import numpy as np
import torch
from .saliency import get_saliency


def prune_factors(head_factor, body_factor, tail_factor, num_filter_keep, criterion='csa'):

    saliency = get_saliency(head_factor, body_factor, tail_factor, criterion)
    ori_num_filter = head_factor.size(2)
    select_index = np.argsort(saliency)[ori_num_filter-num_filter_keep:]
    select_index.sort()

    in_channels = head_factor.size(0)
    rank = head_factor.size(1)
    kernel_size = body_factor.size(0)

    new_head_factor = torch.zeros(in_channels, rank, num_filter_keep)
    new_body_factor = torch.zeros(kernel_size, rank, num_filter_keep)
    new_tail_factor = torch.zeros(kernel_size, rank, num_filter_keep)

    for index_i, i in enumerate(select_index):
        new_head_factor[:, :, index_i] = head_factor[:, :, i]
        new_body_factor[:, :, index_i] = body_factor[:, :, i]
        new_tail_factor[:, :, index_i] = tail_factor[:, :, i]

    return new_head_factor, new_body_factor, new_tail_factor, select_index
