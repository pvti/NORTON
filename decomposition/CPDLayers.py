import torch
import torch.nn as nn


class CPDHead(nn.Module):
    """
    A convolutional HEAD module that implements the Candecomp-Parafac decomposition.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        rank (int): Rank of the weight tensor.
        padding (int): Amount of padding to add to the input tensor.
        weight (torch.Tensor, optional): Weight tensor of shape (in_channels, out_channels, rank).
            If None, the tensor is initialized using Xavier initialization.

    Attributes:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        rank (int): Rank of the weight tensor.
        padding (int): Amount of padding to add to the input tensor.
        weight (torch.Tensor): Weight tensor.

    """

    def __init__(self, in_channels, out_channels, rank, padding, weight=None):
        super(CPDHead, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rank = rank
        self.padding = padding

        if weight is not None:
            assert weight.shape == (
                in_channels, out_channels, rank), "Invalid weight shape"
        else:
            weight = torch.Tensor(in_channels, out_channels, rank)
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
        with torch.no_grad():
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
        padding (int): Amount of padding to add to the input tensor.
        weight (torch.Tensor, optional): Weight tensor of shape (out_channels, rank, kernel_size).
            If None, the tensor is initialized using Xavier initialization.

    Attributes:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        rank (int): Rank of the weight tensor.
        kernel_size (int): Kernel size of the weight tensor.
        padding (int): Amount of padding to add to the input tensor.
        weight (torch.Tensor): Weight tensor.

    """

    def __init__(self, in_channels, out_channels, rank, kernel_size, padding, weight=None):
        super(CPDBody, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rank = rank
        self.kernel_size = kernel_size
        self.padding = padding

        if weight is not None:
            assert weight.shape == (
                out_channels, rank, kernel_size), "Invalid weight shape"
        else:
            weight = torch.Tensor(out_channels, rank, kernel_size)
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
        window_indices = torch.arange(
            w)[:, None] + torch.arange(self.kernel_size)
        Oc_expanded = Oc[:, :, :, :, window_indices]

        # Perform the element-wise multiplication and sum over the last axis (d)
        output = torch.sum(Oc_expanded * B_expanded, dim=-1)

        return output

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
        padding (int): Amount of padding to add to the input tensor.
        weight (torch.Tensor, optional): Weight tensor of shape (out_channels, rank, kernel_size).
            If None, the tensor is initialized using Xavier initialization.
        bias (torch.Tensor, optional): Bias tensor of shape (out_channels,).
            If None, the tensor is initialized to zero.

    Attributes:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        rank (int): Rank of the weight tensor.
        kernel_size (int): Kernel size of the weight tensor.
        padding (int): Amount of padding to add to the input tensor.
        weight (torch.Tensor): Weight tensor.
        bias (torch.Tensor): Bias tensor.

    """

    def __init__(self, in_channels, out_channels, rank, kernel_size, padding, weight=None, bias=None):
        super(CPDTail, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rank = rank
        self.kernel_size = kernel_size
        self.padding = padding

        if weight is not None:
            assert weight.shape == (
                out_channels, rank, kernel_size), "Invalid weight shape"
        else:
            weight = torch.Tensor(out_channels, rank, kernel_size)
            nn.init.xavier_uniform_(weight)

        self.weight = nn.Parameter(weight)

        if bias is not None:
            assert bias.shape == (out_channels,), "Invalid bias shape"
        else:
            bias = torch.zeros(out_channels)

        self.bias = nn.Parameter(bias)

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
        window_indices = torch.arange(
            h)[:, None] + torch.arange(self.kernel_size)
        Ob_expanded = Ob[:, :, :, :, window_indices]

        # Perform the element-wise multiplication and sum over the last axis (d)
        Oa = torch.sum(Ob_expanded * A_expanded, axis=-1)
        Oa = Oa.permute(0, 4, 1, 2, 3)

        # Step 4: Compute O
        output = torch.sum(Oa, dim=-1) + self.bias

        output = output.permute(0, 3, 1, 2)

        return output

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
                f"padding={self.padding})"
                )


if __name__ == "__main__":
    # Test with valid input
    net = nn.Sequential(
        *[CPDHead(64, 128, 9, 1), CPDBody(64, 128, 9, 3, 1), CPDTail(64, 128, 9, 3, 1)])
    print(net)

    # Test with invalid weight shape
    try:
        net = CPDHead(64, 128, 9, 1, weight=torch.rand(5, 5, 5))
    except AssertionError as e:
        raise ValueError(
            "Invalid weight shape. Weight shape should be (out_channels, in_channels, rank).")
