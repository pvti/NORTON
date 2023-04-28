import torch
import torch.nn as nn
from pruning.saliency import get_saliency


class CPDHead(nn.Module):
    """
    A convolutional HEAD module that implements the Candecomp-Parafac decomposition.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        rank (int): Rank of the weight tensor.
        padding (int): Amount of padding to add to the input tensor. Defaults to 1.
        weight (torch.Tensor, optional): Weight tensor of shape (in_channels, out_channels, rank).
            If None, the tensor is initialized using Xavier initialization.

    Attributes:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        rank (int): Rank of the weight tensor.
        padding (int): Amount of padding to add to the input tensor.
        weight (torch.Tensor): Weight tensor.

    """

    def __init__(self, in_channels, out_channels, rank, padding=1, weight=None, device=None):
        super(CPDHead, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rank = rank
        self.padding = padding

        if weight is not None:
            assert weight.shape == (
                in_channels, out_channels, rank), "Invalid weight shape"
        else:
            weight = torch.zeros(
                (in_channels, out_channels, rank), device=device)
            nn.init.xavier_uniform_(weight)

        self.weight = nn.Parameter(weight)

    def forward(self, input):
        """
        Compute the output tensor given an input tensor.

        Args:
            input (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            output (torch.Tensor): Output tensor of shape (batch_size, height + 2*padding, width + 2*padding, out_channels, rank).

        """
        assert input.size(1) == self.in_channels, "Invalid input shape"

        # Add padding to input
        batch_size = input.shape[0]
        padded_I = nn.functional.pad(input, pad=[self.padding]*4)
        padded_I = padded_I.permute(0, 2, 3, 1)

        # Calculate output size after padding
        padded_h = padded_I.shape[1]
        padded_w = padded_I.shape[2]

        # Step 1: Compute Oc
        padded_I_col = padded_I.reshape(
            batch_size * padded_h * padded_w, self.in_channels)
        weight_col = self.weight.reshape(
            self.in_channels, self.out_channels * self.rank)

        # Compute matrix multiplication and reshape output
        output = torch.matmul(padded_I_col, weight_col).reshape(
            batch_size, padded_h, padded_w, self.out_channels, self.rank)

        return output

    def prune(self, selected_index):
        """
        Prunes the filters of the CPDHead based on given indices of filters to keep.

        Args:
            selected_index (List[int]): List of the indices of filters to keep.

        Returns:
            None.
        """
        with torch.no_grad():
            # Update weight tensor and out_channels
            self.weight.set_(self.weight[:, selected_index])
            self.out_channels = len(selected_index)

    def update_in_channels(self, selected_index):
        """
        Updates the number of input channels and weight tensor of the CPDHead
        based on a given list of indices of the filters to keep.

        Args:
            selected_index (List[int]): A list containing the indices of the filters to keep.

        Returns:
            None.
        """
        self.in_channels = len(selected_index)
        with torch.no_grad():
            self.weight.set_(self.weight[selected_index, :])

    def __repr__(self):
        """
        Return a string representation of the module.

        Returns:
            str: String representation of the module.

        """
        return (f"{self.__class__.__name__}("
                f"in_channels={self.in_channels}, "
                f"out_channels={self.out_channels}, "
                f"rank={self.rank}, "
                f"padding={self.padding})"
                )


class CPDBody(nn.Module):
    """
    A convolutional BODY module that implements the Candecomp-Parafac decomposition.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        rank (int): Rank of the weight tensor.
        kernel_size (int): Kernel size of the weight tensor.
        stride (int, optional): Stride. Defaults to 1.
        padding (int): Amount of padding to add to the input tensor. Defaults to 1.
        weight (torch.Tensor, optional): Weight tensor of shape (out_channels, rank, kernel_size).
            If None, the tensor is initialized using Xavier initialization.

    Attributes:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        rank (int): Rank of the weight tensor.
        kernel_size (int): Kernel size of the weight tensor.
        stride (int): Stride.
        padding (int): Amount of padding to add to the input tensor.
        weight (torch.Tensor): Weight tensor.

    """

    def __init__(self, in_channels, out_channels, rank, kernel_size, stride=1, padding=1, weight=None, device=None):
        super(CPDBody, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rank = rank
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        if weight is not None:
            assert weight.shape == (
                out_channels, rank, kernel_size), "Invalid weight shape"
        else:
            weight = torch.zeros(
                (out_channels, rank, kernel_size), device=device)
            nn.init.xavier_uniform_(weight)

        self.weight = nn.Parameter(weight)

    def forward(self, input):
        """
        Compute the output tensor given an input tensor.

        Args:
            input (torch.Tensor): Input tensor of shape (batch_size, height + 2*padding, width + 2*padding, out_channels, rank). Input must be generated from a CPDHead layer.

        Returns:
            output (torch.Tensor): Output tensor of shape (batch_size, height + 2*padding, out_channels, rank, width).

        """

        assert input.shape == (
            input.shape[0], input.shape[1], input.shape[2], self.out_channels, self.rank), "Invalid input shape"

        w = input.size(2) - 2*self.padding

        # Step 2: Compute Ob
        Oc = input.permute(0, 1, 3, 4, 2)
        # Add a new axis to B for broadcasting, B's shape becomes (1, 1, Cout, r, 1, d)
        B_expanded = self.weight[None, None, :, :, None, :]
        # Assuming 'Oc' is a 5-dimensional and 'w' and 'd' are the window width and depth, respectively
        window_indices = torch.arange(start=0, end=w, step=self.stride)[
            :, None] + torch.arange(self.kernel_size)
        Oc_expanded = Oc[:, :, :, :, window_indices]

        # Perform the element-wise multiplication and sum over the last axis (d)
        output = torch.sum(Oc_expanded * B_expanded, dim=-1)

        return output

    def prune(self, selected_index):
        """
        Prunes the filters of the CPDBody based on given indices of filters to keep.

        Args:
            selected_index (List[int]): List of the indices of filters to keep.

        Returns:
            None.
        """
        with torch.no_grad():
            # Update weight tensor and out_channels
            self.weight.set_(self.weight[selected_index, :])
            self.out_channels = len(selected_index)

    def update_in_channels(self, selected_index):
        """
        Updates the number of input channels of the CPDBody based on a given list of indices of the filters to keep.

        Args:
            selected_index (List[int]): A list containing the indices of the filters to keep.

        Returns:
            None.
        """
        self.in_channels = len(selected_index)

    def __repr__(self):
        """
        Return a string representation of the module.

        Returns:
            str: String representation of the module.

        """
        return (f"{self.__class__.__name__}("
                f"in_channels={self.in_channels}, "
                f"out_channels={self.out_channels}, "
                f"rank={self.rank}, "
                f"kernel_size={self.kernel_size}, "
                f"stride={self.stride}, "
                f"padding={self.padding})"
                )


class CPDTail(nn.Module):
    """
    A convolutional TAIL module that implements the Candecomp-Parafac decomposition.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        rank (int): Rank of the weight tensor.
        kernel_size (int): Kernel size of the weight tensor.
        stride (int, optional): Stride. Defaults to 1.
        padding (int): Amount of padding to add to the input tensor. Defaults to 1.
        weight (torch.Tensor, optional): Weight tensor of shape (out_channels, rank, kernel_size).
            If None, the tensor is initialized using Xavier initialization.
        biased (bool, optional): Whether to include a bias term. Default is True.
        bias (torch.Tensor, optional): Bias tensor of shape (out_channels,).
            If None, the tensor is initialized to zero.

    Attributes:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        rank (int): Rank of the weight tensor.
        kernel_size (int): Kernel size of the weight tensor.
        stride (int): Stride.
        padding (int): Amount of padding to add to the input tensor.
        weight (torch.Tensor): Weight tensor.
        bias (torch.Tensor or None): Bias tensor. If biased=False, this will be None.

    """

    def __init__(self, in_channels, out_channels, rank, kernel_size, stride=1, padding=1, weight=None, biased=True, bias=None, device=None):
        super(CPDTail, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rank = rank
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.biased = biased

        if weight is not None:
            assert weight.shape == (
                out_channels, rank, kernel_size), "Invalid weight shape"
        else:
            weight = torch.zeros(
                (out_channels, rank, kernel_size), device=device)
            nn.init.xavier_uniform_(weight)

        self.weight = nn.Parameter(weight)

        if biased:
            if bias is not None:
                assert bias.shape == (out_channels,), "Invalid bias shape"
            else:
                bias = torch.zeros(out_channels, device=device)
            self.bias = nn.Parameter(bias)
        else:
            self.bias = None

    def forward(self, input):
        """
        Compute the output tensor given an input tensor.

        Args:
            input (torch.Tensor): Input tensor of shape (batch_size, height + 2*padding, out_channels, rank, width). Input must be generated from a CPDBody layer.

        Returns:
            output (torch.Tensor): Output tensor of shape (batch_size, out_channels, height, width).

        """

        assert input.shape == (
            input.shape[0], input.shape[1], self.out_channels, self.rank, input.shape[4]), "Invalid input shape"

        h = input.size(1) - 2*self.padding

        # Step 3: Compute Oa
        Ob = input.permute(0, 4, 2, 3, 1)
        # Add a new axis to B for broadcasting, A's shape becomes (1, 1, Cout, r, 1, d)
        A_expanded = self.weight[None, None, :, :, None, :]

        # Assuming 'Ob' is a 5-dimensional and 'h' and 'd' are the window width and depth, respectively
        window_indices = torch.arange(start=0, end=h, step=self.stride)[
            :, None] + torch.arange(self.kernel_size)
        Ob_expanded = Ob[:, :, :, :, window_indices]

        # Perform the element-wise multiplication and sum over the last axis (d)
        Oa = torch.sum(Ob_expanded * A_expanded, axis=-1)
        Oa = Oa.permute(0, 4, 1, 2, 3)

        # Step 4: Compute O
        output = torch.sum(Oa, dim=-1)

        if self.biased:
            output += self.bias

        output = output.permute(0, 3, 1, 2)

        return output

    def prune(self, selected_index):
        """
        Prunes the filters of the CPDTail based on a given number of filters to keep.

        Args:
            n_keep (int): Number of filters to keep.

        Returns:
        """
        with torch.no_grad():
            # Update weight, bias tensor and out_channels
            self.weight.set_(self.weight[selected_index, :])
            if self.biased:
                self.bias.set_(self.bias[selected_index])
            self.out_channels = len(selected_index)

    def update_in_channels(self, selected_index):
        """
        Updates the number of input channels of the CPDTail based on given indices of filters to keep.

        Args:
            selected_index (List[int]): List of the indices of filters to keep.

        Returns:
            None.
        """
        self.in_channels = len(selected_index)

    def __repr__(self):
        """
        Return a string representation of the module.

        Returns:
            str: String representation of the module.

        """
        return (f"{self.__class__.__name__}("
                f"in_channels={self.in_channels}, "
                f"out_channels={self.out_channels}, "
                f"rank={self.rank}, "
                f"kernel_size={self.kernel_size}, "
                f"stride={self.stride}, "
                f"padding={self.padding}, "
                f"biased={self.biased})"
                )


class CPDLayer(nn.Sequential):
    """
    A wrapper layer for the Candecomp-Parafac decomposition (CPD) that applies a CPD-based
    convolution to an input tensor.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        rank (int): Rank of the weight tensor used in the CPD decomposition.
        kernel_size (int, optional): Size of the convolution kernel. Defaults to 3.
        stride (int, optional): Stride. Defaults to 1.
        padding (int, optional): Amount of padding to add to the input tensor. Defaults to 1.
        head_weight (torch.Tensor): Weight tensor for CPDHead.
        body_weight (torch.Tensor): Weight tensor for CPDBody.
        tail_weight (torch.Tensor): Weight tensor for CPDTail.
        biased (bool, optional): Whether to include a bias term. Default is True.
        bias (torch.Tensor, optional): Bias tensor for CPDTail.

    Attributes:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        rank (int): Rank of the weight tensor.
        kernel_size (int): Kernel size of the weight tensor.
        stride (int): Stride.
        padding (int): Amount of padding to add to the input tensor.
        head (CPDHead): The head module of the CPD-based convolution layer.
        body (CPDBody): The body module of the CPD-based convolution layer.
        tail (CPDTail): The tail module of the CPD-based convolution layer.

    """

    def __init__(self, in_channels, out_channels, rank, kernel_size=3, stride=1, padding=1, head_weight=None, body_weight=None, tail_weight=None, biased=True, bias=None, device=None):
        super(CPDLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rank = rank
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.biased = biased
        self.device = device

        self.add_module("head", CPDHead(in_channels, out_channels,
                        rank, padding, head_weight, device))
        self.add_module("body", CPDBody(in_channels, out_channels,
                        rank, kernel_size, stride, padding, body_weight, device))
        self.add_module("tail", CPDTail(in_channels, out_channels,
                        rank, kernel_size, stride, padding, tail_weight, biased, bias, device))

    def prune(self, compress_rate, criterion):
        """
        Prunes the weights of each sublayer of the CPDLayer based on a given compression rate.

        Args:
            compress_rate (float): A value between 0 and 1 indicating the percentage of filters to prune.

        Returns:
            selected_index (Tuple[List[int], List[int], List[int]]): A tuple containing the index of the filters to keep
            in the head, body, and tail sublayers respectively.
        """
        n_keep = int(self.out_channels * (1 - compress_rate))
        self.out_channels = n_keep
        with torch.no_grad():
            head_weight = self.head.weight.detach().clone()
            head_weight = head_weight.transpose(1, 0)
        saliency = get_saliency(
            head_weight, self.body.weight, self.tail.weight, criterion)
        _, selected_index = torch.topk(saliency, n_keep)
        self.head.prune(selected_index)
        self.body.prune(selected_index)
        self.tail.prune(selected_index)

        return selected_index

    def update_in_channels(self, selected_index):
        """
        Update the self.in_channels to length of selected_index and call the update_in_channels method of the sub-layers.

        Args:
            selected_index (List[int]): A list containing the index of the filters to keep
            in the head sublayer.
        """
        self.in_channels = len(selected_index)
        self.head.update_in_channels(selected_index)
        self.body.update_in_channels(selected_index)
        self.tail.update_in_channels(selected_index)


if __name__ == "__main__":
    net = CPDLayer(64, 128, 9, 3, 1)
    print(net)
