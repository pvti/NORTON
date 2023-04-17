import torch
import torch.nn as nn


def prune_bn_module(bn_module: nn.BatchNorm2d, selected_indices: torch.Tensor):
    # Get the number of selected channels
    num_selected_channels = len(selected_indices)

    # Create new tensors for the weights and biases based on the selected indices
    new_weight = nn.Parameter(bn_module.weight[selected_indices])
    new_bias = nn.Parameter(bn_module.bias[selected_indices])

    # Assign the new weights and biases to the BatchNorm2d instance
    bn_module.weight = new_weight
    bn_module.bias = new_bias

    # If running_mean and running_var are also present, update them as well
    if bn_module.running_mean is not None and bn_module.running_var is not None:
        # Get the current number of channels
        num_channels = bn_module.running_mean.size(0)

        # Compute the mean and variance for the selected channels
        selected_mean = bn_module.running_mean[selected_indices]
        selected_var = bn_module.running_var[selected_indices]

        # Compute the scaling factor for the variance
        scale_factor = num_channels / num_selected_channels

        # Compute the new running mean and variance
        new_running_mean = selected_mean
        new_running_var = selected_var * scale_factor

        # Assign the new running mean and variance to the BatchNorm2d instance
        bn_module.running_mean = new_running_mean
        bn_module.running_var = new_running_var

    # Update the number of input channels in the BatchNorm2d instance
    bn_module.num_features = num_selected_channels

    return bn_module
