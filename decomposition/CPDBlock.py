import torch
import torch.nn as nn
from collections import OrderedDict


class CPDBlock(nn.Module):
    """
    A convolutional module that implements the Candecomp-Parafac filter decomposition.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        rank (int): Rank of the weight tensor used in the CPD decomposition.
        kernel_size (int, optional): Size of the convolution kernel. Defaults to 3.
        stride (int, optional): Stride. Defaults to 1.
        padding (int, optional): Amount of padding to add to the input tensor. Defaults to 1.
        bias (bool, optional): If True, adds a learnable bias to the output. Default: True

    Attributes:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        rank (int): Rank of the weight tensor.
        kernel_size (int): Kernel size of the weight tensor.
        stride (int): Stride.
        padding (int): Amount of padding to add to the input tensor.
        pointwise (Conv2d): The pointwise module of the CPD-based convolution layer.
        vertical (Conv2d): The vertical module of the CPD-based convolution layer.
        horizontal (Conv2d): The horizontal module of the CPD-based convolution layer.

    """

    def __init__(self, in_channels, out_channels, rank, kernel_size=3, stride=1, padding=1, bias=True, device=None):
        super(CPDBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rank = rank
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.device = device

        channels = rank*out_channels

        self.feature = nn.Sequential(OrderedDict([
            ('pointwise', nn.Conv2d(in_channels, channels, 1, stride=1, padding=0, bias=False)),
            ('vertical', nn.Conv2d(channels, channels, kernel_size=(kernel_size, 1),
                                   stride=(stride, 1), padding=(padding, 0), groups=channels, bias=False)),
            ('horizontal', nn.Conv2d(channels, channels, kernel_size=(1, kernel_size),
                                     stride=(1, stride), padding=(0, padding), groups=channels, bias=False))
        ]))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels, device=device))
        else:
            self.bias = None

    def forward(self, input):
        """
        Compute the output tensor given an input tensor.

        Args:
            input (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            output (torch.Tensor): Output tensor of shape (batch_size, out_channels, height, width).

        """
        output = self.feature(input)

        output = output.view(
            output.shape[0], self.rank, self.out_channels, *output.shape[2:])
        output = torch.sum(output, dim=1)

        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1)

        return output


if __name__ == "__main__":
    net = CPDBlock(64, 128, 9, 3, 1)
    print(net)
