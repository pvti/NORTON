import torch
from scipy.linalg import subspace_angles
import numpy as np


def get_saliency(head_weight, body_weight, tail_weight):
    """
    Computes the saliency of each filter in the weight tensor.

    Args:
        head_weight (torch.Tensor): The weight tensor of a CPDHead layer, of shape (out_channels, in_channels, rank).
        head_weight (torch.Tensor): The weight tensor of a CPDBody layer, of shape (out_channels, rank, kernel_size).
        head_weight (torch.Tensor): The weight tensor of a CPDTail layer, of shape (out_channels, rank, kernel_size).

    Returns:
        saliency (torch.Tensor): A 1D tensor containing the saliency of each filter.
    """
    num_filters = head_weight.shape[0]
    distance_matrix = torch.zeros(num_filters, num_filters)
    for i in range(num_filters-1):
        for j in range(i+1, num_filters):
            head = VBD(head_weight[i], head_weight[j])
            body = VBD(body_weight[i], body_weight[j])
            tail = VBD(tail_weight[i], tail_weight[j])
            distance_matrix[i, j] = head*body*tail
            distance_matrix[j, i] = distance_matrix[i, j]

    saliency = torch.full((num_filters, ), num_filters-1)
    inf = float('inf')
    distance_matrix.fill_diagonal_(inf)
    for i in range(num_filters-1):
        pos = torch.where(distance_matrix == torch.min(distance_matrix))[0] # since distance_matrix is symmetrix, get only d[i,j] or d[j,i]
        row, col = pos[0].item(), pos[1].item()
        # Compute the sum of the distance of filter `row` to other filters (excluding `inf` value)
        row_sum = torch.sum(distance_matrix[row][distance_matrix[row] != inf])
        # Compute the sum of the distance of filter `col` to other filters (excluding `inf` value)
        col_sum = torch.sum(distance_matrix[:, col][distance_matrix[:, col] != inf])

        # Choose the filter with smaller sum of distances as the less important filter
        index = row if row_sum < col_sum else col

        # Update saliency array and set the distances of the selected filter to `inf`
        saliency[i] = index
        distance_matrix[index, :] = distance_matrix[:, index] = inf

    # # subspace
    # similarity_matrix = torch.zeros(num_filters, num_filters)
    # for i in range(num_filters-1):
    #     for j in range(i+1, num_filters):
    #         head = CSA(head_weight[i], head_weight[j])
    #         body = CSA(body_weight[i], body_weight[j])
    #         tail = CSA(tail_weight[i], tail_weight[j])
    #         similarity_matrix[i, j] = head*body*tail
    #         similarity_matrix[j, i] = similarity_matrix[i, j]

    # saliency = torch.full((num_filters, ), num_filters-1)
    # inf = -float('inf')
    # similarity_matrix.fill_diagonal_(inf)
    # for i in range(num_filters-1):
    #     pos = torch.where(similarity_matrix == torch.max(similarity_matrix))[0] # since distance_matrix is symmetrix, get only d[i,j] or d[j,i]
    #     row, col = pos[0].item(), pos[1].item()
    #     # Compute the sum of the distance of filter `row` to other filters (excluding `inf` value)
    #     row_sum = torch.sum(similarity_matrix[row][similarity_matrix[row] != inf])
    #     # Compute the sum of the distance of filter `col` to other filters (excluding `inf` value)
    #     col_sum = torch.sum(similarity_matrix[:, col][similarity_matrix[:, col] != inf])

    #     # Choose the filter with smaller sum of distances as the less important filter
    #     index = row if row_sum < col_sum else col

    #     # Update saliency array and set the distances of the selected filter to `inf`
    #     saliency[i] = index
    #     similarity_matrix[index, :] = similarity_matrix[:, index] = inf

    return saliency


def VBD(x, y):
    """Caculate variance based distance
    """
    vbd = torch.var(x-y) / (torch.var(x) + torch.var(y))

    return vbd


def CSA(x, y):
    """
    """
    np_x = x.detach().cpu().numpy()
    np_y = y.detach().cpu().numpy()
    principal_angles = subspace_angles(np_x, np_y)
    csa = np.cos(principal_angles[-1])**2

    return csa
