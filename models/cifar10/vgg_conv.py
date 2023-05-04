import torch
import torch.nn as nn
from collections import OrderedDict


class CPDBasicBlock(nn.Sequential):
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

    def __init__(self, in_channels, kernel_size=3, stride=1, padding=1, bias=True):
        super(CPDBasicBlock, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.add_module('head', )
        self.add_module('body', )
        self.add_module('tail', )


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

    def __init__(self, in_channels, out_channels, rank, kernel_size=3, stride=1, padding=1, bias=True):
        super(CPDBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rank = rank
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # branch conv2d's output height and weight must be identical to input's
        # self.branches = nn.ModuleList([CPDBasicBlock(in_channels, kernel_size, stride, padding, bias) for _ in range(out_channels)])
        list_branch = []
        for i in range(self.out_channels):
            branch = nn.Sequential(
                nn.Conv2d(in_channels, 1, 1, padding=0, bias=False),
                nn.Conv2d(1, 1, kernel_size=(1, kernel_size), padding=(0, padding), bias=False),
                nn.Conv2d(1, 1, kernel_size=(kernel_size, 1), padding=(padding, 0), bias=bias)
            )
            list_branch.append(branch)
        self.branches = nn.ModuleList(list_branch)

    def forward(self, input):
        """
        Compute the output tensor given an input tensor.

        Args:
            input (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            output (torch.Tensor): Output tensor of shape (batch_size, out_channels, height, width).

        """
        # output = torch.empty((input.shape[0], self.out_channels, input.shape[2], input.shape[3]), dtype=input.dtype, device=input.device)
        # for i in range(self.out_channels):
        #     output[:, i] = self.branches[i](input)[:, 0]

        # return output
        output = torch.stack([self.branches[i](input)[:, 0] for i in range(self.out_channels)], dim=1)

        return output.to(input.device, input.dtype)



defaultcfg = [64, 64, 'M', 128, 128, 'M', 256, 256,
              256, 'M', 512, 512, 512, 'M', 512, 512, 512]


class VGG(nn.Module):
    def __init__(self, compress_rate=[0.0]*12, rank=0, cfg=None, num_classes=10):
        super(VGG, self).__init__()

        if cfg is None:
            cfg = defaultcfg

        self.compress_rate = compress_rate[:]
        self.compress_rate.append(0.0)

        self.rank = rank

        self.features = self._make_layers(cfg)
        self.classifier = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(cfg[-2], cfg[-1])),
            ('norm1', nn.BatchNorm1d(cfg[-1])),
            ('relu1', nn.ReLU(inplace=True)),
            ('linear2', nn.Linear(cfg[-1], num_classes)),
        ]))

    def _make_layers(self, cfg):

        layers = nn.Sequential()
        in_channels = 3
        cnt = 0

        for i, x in enumerate(cfg):
            if x == 'M':
                layers.add_module('pool%d' %
                                  i, nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                x = int(x * (1-self.compress_rate[cnt]))
                cnt += 1
                layer = None
                if self.rank > 0:
                    layer = CPDBlock(in_channels, x, self.rank,
                                     kernel_size=3, padding=1)
                else:
                    layer = nn.Conv2d(in_channels, x, kernel_size=3, padding=1)
                layers.add_module('conv%d' % i, layer)
                layers.add_module('norm%d' % i, nn.BatchNorm2d(x))
                layers.add_module('relu%d' % i, nn.ReLU(inplace=True))
                in_channels = x

        return layers

    def forward(self, x):
        x = self.features(x)
        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x


def vgg_16_bn(compress_rate=[0.0]*12, rank=0):
    """
    A custom VGG-16-BN module with compression rate and rank for CPD.

    Args:
        compress_rate (list[int]): Compression rate. Default to 0 for all layers.
        rank (int): Rank for CPD. Defaults to 0.
            If 0, use the Conv2d, else use CPDLayer.

    Returns:
        A VGG object.

    """

    return VGG(compress_rate=compress_rate, rank=rank)
