import numpy as np
from fancy_einsum import einsum
from einops import reduce, rearrange, repeat
from typing import Union, Optional, Callable
import torch as t
import functools
from torch import nn

IntOrPair = Union[int, tuple[int, int]]
Pair = tuple[int, int]


def force_pair(v: IntOrPair) -> Pair:
    """Convert v to a pair of int, if it isn't already."""
    if isinstance(v, tuple):
        if len(v) != 2:
            raise ValueError(v)
        return (int(v[0]), int(v[1]))
    elif isinstance(v, int):
        return (v, v)
    raise ValueError(v)


def pad1d(x: t.Tensor, left: int, right: int, pad_value: float) -> t.Tensor:
    """Return a new tensor with padding applied to the edges.

    x: shape (batch, in_channels, width), dtype float32

    Return: shape (batch, in_channels, left + right + width)
    """
    B, IC, W = x.shape
    new_W = W + left + right
    padded_x = x.new_full(size=(B, IC, new_W), fill_value=pad_value)
    padded_x[:, :, left : left + W] = x
    return padded_x


def pad2d(
    x: t.Tensor, left: int, right: int, top: int, bottom: int, pad_value: float
) -> t.Tensor:
    """Return a new tensor with padding applied to the edges.

    x: shape (batch, in_channels, height, width), dtype float32

    Return: shape (batch, in_channels, top + height + bottom, left + width + right)
    """
    B, IC, H, W = x.shape
    new_H = H + top + bottom
    new_W = W + left + right
    padded_x = x.new_full(size=(B, IC, new_H, new_W), fill_value=pad_value)
    padded_x[..., top : top + H, left : left + W] = x
    return padded_x


def conv1d(x, weights, stride: int = 1, padding: int = 0) -> t.Tensor:
    """Like torch's conv1d using bias=False.

    x: shape (batch, in_channels, width)
    weights: shape (out_channels, in_channels, kernel_width)

    Returns: shape (batch, out_channels, output_width)
    """
    x = pad1d(x, left=padding, right=padding, pad_value=0)

    batch, in_channels, width = x.shape
    kernel_width = weights.shape[2]
    output_width = 1 + (width - kernel_width) // stride

    batch_stride, input_stride, width_stride = x.stride()

    x_strided = x.as_strided(
        size=(batch, in_channels, output_width, kernel_width),
        stride=(batch_stride, input_stride, width_stride * stride, width_stride),
    )

    return einsum("b ic ow kw, oc ic kw -> b oc ow", x_strided, weights)


def conv2d(x, weights, stride: IntOrPair = 1, padding: IntOrPair = 0) -> t.Tensor:
    """Like torch's conv2d using bias=False

    x: shape (batch, in_channels, height, width)
    weights: shape (out_channels, in_channels, kernel_height, kernel_width)


    Returns: shape (batch, out_channels, output_height, output_width)
    """
    stride, padding = force_pair(stride), force_pair(padding)

    x = pad2d(
        x,
        left=padding[1],
        right=padding[1],
        top=padding[0],
        bottom=padding[0],
        pad_value=0,
    )

    batch, in_channels, height, width = x.shape
    _, _, k_height, k_width = weights.shape

    output_height = 1 + (height - k_height) // stride[0]
    output_width = 1 + (width - k_width) // stride[1]

    batch_stride, input_stride, height_stride, width_stride = x.stride()

    x_strided = x.as_strided(
        size=(batch, in_channels, output_height, output_width, k_height, k_width),
        stride=(
            batch_stride,
            input_stride,
            height_stride * stride[0],
            width_stride * stride[1],
            height_stride,
            width_stride,
        ),
    )

    return einsum("b ic oh ow kh kw, oc ic kh kw -> b oc oh ow", x_strided, weights)


def maxpool2d(
    x: t.Tensor,
    kernel_size: IntOrPair,
    stride: Optional[IntOrPair] = None,
    padding: IntOrPair = 0,
) -> t.Tensor:
    """Like PyTorch's maxpool2d.

    x: shape (batch, channels, height, width)
    stride: if None, should be equal to the kernel size

    Return: (batch, channels, out_height, output_width)
    """
    if stride is None:
        stride = kernel_size
    stride, padding, kernel = (
        force_pair(stride),
        force_pair(padding),
        force_pair(kernel_size),
    )

    x = pad2d(
        x,
        left=padding[1],
        right=padding[1],
        top=padding[0],
        bottom=padding[0],
        pad_value=-t.inf,
    )

    batch, in_channels, height, width = x.shape
    output_height = 1 + (height - kernel[0]) // stride[0]
    output_width = 1 + (width - kernel[1]) // stride[1]

    batch_stride, input_stride, height_stride, width_stride = x.stride()

    x_strided = x.as_strided(
        size=(batch, in_channels, output_height, output_width, kernel[0], kernel[1]),
        stride=(
            batch_stride,
            input_stride,
            height_stride * stride[0],
            width_stride * stride[1],
            height_stride,
            width_stride,
        ),
    )

    return t.amax(x_strided, dim=(-1, -2))


class MaxPool2d(nn.Module):
    def __init__(
        self,
        kernel_size: IntOrPair,
        stride: Optional[IntOrPair] = None,
        padding: IntOrPair = 1,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: t.Tensor) -> t.Tensor:
        """Call the functional version of maxpool2d."""
        x = maxpool2d(x, self.kernel_size, self.stride, self.padding)
        return x

    def extra_repr(self) -> str:
        """Add additional information to the string representation of this class."""
        return f"Kernel size: {self.kernel_size} Stride: {self.stride} Padding: {self.padding}"


class ReLU(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        return t.maximum(x, t.tensor(0.0))


class Flatten(nn.Module):
    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input: t.Tensor) -> t.Tensor:
        """Flatten out dimensions from start_dim to end_dim, inclusive of both."""
        t_dims = input.shape
        end = self.end_dim if self.end_dim >= 0 else len(t_dims) + self.end_dim
        flattened_size = functools.reduce(
            lambda x, y: x * y, t_dims[self.start_dim : end + 1]
        )

        new_shape = t_dims[: self.start_dim] + (flattened_size,) + t_dims[end + 1 :]

        return t.reshape(input, new_shape)

    def extra_repr(self) -> str:
        return f"Reshapes from dim {self.start_dim} to {self.end_dim} inclusive"


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias=True):
        """A simple linear (technically, affine) transformation.

        The fields should be named `weight` and `bias` for compatibility with PyTorch.
        If `bias` is False, set `self.bias` to None.
        """
        super().__init__()
        k = 1 / np.sqrt(in_features)

        self.weight = nn.Parameter(
            t.zeros(out_features, in_features).uniform_(-k, to=k)
        )
        self.bias = (
            None if not bias else nn.Parameter(t.zeros(out_features).uniform_(-k, to=k))
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        x: shape (*, in_features)
        Return: shape (*, out_features)
        """
        out = einsum("... in_f, out_f in_f -> ... out_f", x, self.weight)
        if self.bias != None:
            out += self.bias

        return out

    def extra_repr(self) -> str:
        pass


class Conv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: IntOrPair,
        stride: IntOrPair = 1,
        padding: IntOrPair = 0,
    ):
        """
        Same as torch.nn.Conv2d with bias=False.

        Name your weight field `self.weight` for compatibility with the PyTorch version.
        """
        super().__init__()
        self.stride = stride
        self.padding = padding
        kernel_height, kernel_width = force_pair(kernel_size)
        k = 1 / np.sqrt(in_channels * kernel_height * kernel_width)

        self.weight = nn.Parameter(
            t.zeros(out_channels, in_channels, kernel_height, kernel_width).uniform_(
                -k, to=k
            )
        )
        self.bias = None  # if not bias else nn.Parameter(t.zeros(out_channels).uniform_(-k, to=k))

    def forward(self, x: t.Tensor) -> t.Tensor:
        """Apply the functional conv2d you wrote earlier."""
        out = conv2d(x, self.weight, stride=self.stride, padding=self.padding)
        return out

    def extra_repr(self) -> str:
        pass
