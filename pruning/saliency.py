import torch


def get_saliency(weight):
    """
    Computes the saliency of each filter in the weight tensor.

    Args:
        weight (torch.Tensor): The weight tensor of shape (out_channels, W, H).

    Returns:
        saliency (torch.Tensor): A 1D tensor containing the saliency of each filter.
    """
    num_filters = weight.shape[0]
    distance_matrix = torch.zeros(num_filters, num_filters)
    for i in range(num_filters-1):
        for j in range(i+1, num_filters):
            distance_matrix[i, j] = VBD(weight[i], weight[j])
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

    return saliency

def VBD(x, y):
    """Caculate variance based distance
    """
    vbd = torch.var(x-y) / (torch.var(x) + torch.var(y))

    return vbd
    