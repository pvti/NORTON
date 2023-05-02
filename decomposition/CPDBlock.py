import torch
import torch.nn as nn


class CPDBlock(nn.Module):
    """
    A convolutional module that implements the Candecomp-Parafac decomposition.

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
        head (ModuleList): The head module of the CPD-based convolution layer.
        body (Conv2d): The body module of the CPD-based convolution layer.
        tail (Conv2d): The tail module of the CPD-based convolution layer.

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

        # branch conv2d's output height and weight must be identical to input's
        self.head = nn.ModuleList([nn.Conv2d(in_channels,
                                             out_channels,
                                             kernel_size=1,
                                             stride=1,
                                             padding=0,
                                             bias=False,
                                             device=device)
                                   for _ in range(rank)])
        self.body = nn.Conv2d(out_channels,
                              rank,
                              kernel_size=(kernel_size, 1),
                              stride=(stride, 1),
                              padding=(padding, 0),
                              bias=False,
                              device=device)
        self.tail = nn.Conv2d(rank,
                              out_channels,
                              kernel_size=(1, kernel_size),
                              stride=(1, stride),
                              padding=(0, padding),
                              bias=bias,
                              device=device)

    def forward(self, input):
        """
        Compute the output tensor given an input tensor.

        Args:
            input (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            output (torch.Tensor): Output tensor of shape (batch_size, out_channels, height, width).

        """
        assert input.shape[1] == self.in_channels, "Invalid input shape"

        batch_size = input.shape[0]
        height = input.shape[2]
        width = input.shape[3]

        output = torch.empty((batch_size, self.out_channels, height,
                             width, self.rank), dtype=input.dtype, device=input.device)
        for i in range(self.rank):
            output[:, :, :, :, i] = self.head[i](input)
        output = output.sum(dim=-1)

        output = self.body(output)

        output = self.tail(output)

        return output


if __name__ == "__main__":
    net = CPDBlock(64, 128, 9, 3, 1)
    print(net)
