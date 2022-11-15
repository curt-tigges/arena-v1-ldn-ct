import numpy as np
from fancy_einsum import einsum
from einops import reduce, rearrange, repeat
from typing import Union, Optional, Callable
import torch as t
import functools
from torch import nn


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



