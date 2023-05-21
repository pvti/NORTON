import torch
from tqdm.auto import tqdm
import math


pi = torch.tensor(math.pi, device=0)


def get_saliency(head_factor, body_factor, tail_factor, criterion='pabs'):
    """
    Computes the saliency of each filter in the factor based on a criterion.

    Args:
        head_factor (torch.Tensor): The factor tensor of a CPDHead layer, of shape (in_channels, rank, out_channels).
        body_factor (torch.Tensor): The factor tensor of a CPDBody layer, of shape (kernel_size, rank, out_channels).
        tail_factor (torch.Tensor): The factor tensor of a CPDTail layer, of shape (kernel_size, rank, out_channels).
        criterion (srt): vbd or csa

    Returns:
        saliency (torch.Tensor): A 1D tensor containing the saliency of each filter.
    """
    num_filters = head_factor.size(2)
    saliency = torch.full((num_filters, ), num_filters-1)

    if criterion == 'vbd':
        distance_matrix = torch.zeros(num_filters, num_filters)
        for i in tqdm(range(num_filters-1)):
            for j in range(i+1, num_filters):
                head = VBD(head_factor[:, :, i], head_factor[:, :, j])
                # body = VBD(body_factor[:, :, i], body_factor[:, :, j])
                # tail = VBD(tail_factor[:, :, i], tail_factor[:, :, j])
                distance_matrix[i, j] = head#*body*tail
                distance_matrix[j, i] = distance_matrix[i, j]

        inf = float('inf')
        distance_matrix.fill_diagonal_(inf)
        for i in range(num_filters-1):
            # since distance_matrix is symmetrix, get only d[i,j] or d[j,i]
            pos = torch.where(distance_matrix == torch.min(distance_matrix))[0]
            row, col = pos[0].item(), pos[1].item()
            # Compute the sum of the distance of filter `row` to other filters (excluding `inf` value)
            row_sum = torch.sum(
                distance_matrix[row][distance_matrix[row] != inf])
            # Compute the sum of the distance of filter `col` to other filters (excluding `inf` value)
            col_sum = torch.sum(
                distance_matrix[:, col][distance_matrix[:, col] != inf])

            # Choose the filter with smaller sum of distances as the less important filter
            index = row if row_sum < col_sum else col

            # Update saliency array and set the distances of the selected filter to `inf`
            saliency[i] = index
            distance_matrix[index, :] = distance_matrix[:, index] = inf

    elif criterion == 'csa':
        similarity_matrix = torch.zeros(num_filters, num_filters)
        # for i in tqdm(range(num_filters-1)):
        for i in tqdm(range(num_filters)):  # so that tqdm shows num_filters
            for j in range(i+1, num_filters):
                head = CSA(head_factor[:, :, i], head_factor[:, :, j])
                # body = CSA(body_factor[:, :, i], body_factor[:, :, j])
                # tail = CSA(tail_factor[:, :, i], tail_factor[:, :, j])
                similarity_matrix[i, j] = head#*body*tail
                similarity_matrix[j, i] = similarity_matrix[i, j]

        inf = -float('inf')
        similarity_matrix.fill_diagonal_(inf)
        for i in range(num_filters-1):
            # since distance_matrix is symmetrix, get only d[i,j] or d[j,i]
            pos = torch.where(similarity_matrix ==
                              torch.max(similarity_matrix))[0]
            row, col = pos[0].item(), pos[1].item()
            # Compute the sum of the distance of filter `row` to other filters (excluding `inf` value)
            row_sum = torch.sum(
                similarity_matrix[row][similarity_matrix[row] != inf])
            # Compute the sum of the distance of filter `col` to other filters (excluding `inf` value)
            col_sum = torch.sum(
                similarity_matrix[:, col][similarity_matrix[:, col] != inf])

            # Choose the filter with smaller sum of distances as the less important filter
            index = row if row_sum < col_sum else col

            # Update saliency array and set the distances of the selected filter to `inf`
            saliency[i] = index
            similarity_matrix[index, :] = similarity_matrix[:, index] = inf

    elif criterion == 'pabs':
        distance_matrix = torch.zeros(num_filters, num_filters)
        for i in tqdm(range(num_filters)):
            for j in range(i+1, num_filters):
                head = subspace_angles(head_factor[:, :, i], head_factor[:, :, j])
                body = subspace_angles(body_factor[:, :, i], body_factor[:, :, j])
                tail = subspace_angles(tail_factor[:, :, i], tail_factor[:, :, j])
                distance_matrix[i, j] = head + body + tail
                distance_matrix[j, i] = distance_matrix[i, j]

        inf = float('inf')
        distance_matrix.fill_diagonal_(inf)
        for i in range(num_filters-1):
            # since distance_matrix is symmetrix, get only d[i,j] or d[j,i]
            pos = torch.where(distance_matrix == torch.min(distance_matrix))[0]
            row, col = pos[0].item(), pos[1].item()
            # Compute the sum of the distance of filter `row` to other filters (excluding `inf` value)
            row_sum = torch.sum(
                distance_matrix[row][distance_matrix[row] != inf])
            # Compute the sum of the distance of filter `col` to other filters (excluding `inf` value)
            col_sum = torch.sum(
                distance_matrix[:, col][distance_matrix[:, col] != inf])

            # Choose the filter with smaller sum of distances as the less important filter
            index = row if row_sum < col_sum else col

            # Update saliency array and set the distances of the selected filter to `inf`
            saliency[i] = index
            distance_matrix[index, :] = distance_matrix[:, index] = inf

    return saliency


def VBD(x, y):
    """Caculate variance based distance
    """
    vbd = torch.var(x-y) / (torch.var(x) + torch.var(y))

    return vbd


def CSA(x, y):
    """
    """
    csa = torch.cos(subspace_angles(x, y))**2

    return csa


def subspace_angles(A, B):
    A, _ = torch.linalg.qr(A)
    B, _ = torch.linalg.qr(B)
    if A.size(1) < B.size(1):
        A, B = B, A
    B = B - torch.matmul(A, torch.matmul(A.transpose(1, 0), B))
    theta = torch.asin(torch.min(torch.tensor(1.0), torch.norm(B)))

    theta = torch.abs(theta)
    theta = torch.remainder(theta, pi/2)

    return theta


def get_saliency_conv(weight):
    """
    Computes the saliency of each filter in 1x1 convolution of resnet50 based on its norm.

    Args:
        weight (torch.Tensor): The weight tensor of a convolution layer, of shape (out_channels, in_channels, 1, 1).

    Returns:
        saliency (torch.Tensor): A 1D tensor containing the saliency of each filter.
    """
    num_filters = weight.size(0)
    saliency = []
    for i in tqdm(range(num_filters)):
        filter = weight[i]
        saliency.append(torch.norm(filter).view(1))

    return torch.cat(saliency)
