from inspect import signature
import math
from functools import partial
from abc import abstractmethod
from collections import OrderedDict
from typing import Any, Union, Self

import lightning as L
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from einops._torch_specific import allow_ops_in_compiled_graph

from models.config import modelConfig
from models.utils import parse_lightning_ckpt
from models.networks.modules import Any, modelConfig, nn  # requires einops>=0.6.1
import logging

allow_ops_in_compiled_graph()  # For torch.compile compatibility

import os
import sys
import time
import traceback
from pathlib import Path


class NaNDetectedError(RuntimeError):
    """Raised when a NaN is detected during module forward/processing."""

    pass


class configModuleBaseLightning(L.LightningModule):
    """
    A base class for modules that use a configuration object.

    This class provides a class method to construct an instance of the class
    using a configuration object.
    """

    @classmethod
    def from_config(cls: type, config: modelConfig, *args: Any, **kwargs: Any) -> Self:
        """
        Construct a neural network module from a given configuration.

        Parameters
        ----------
        cls : type
            The class of the neural network module.
        config : modelConfig
            The configuration object containing the necessary parameters for constructing the module.
        *args : Any
            Additional positional arguments to be passed to the constructor of the module.
        **kwargs : Any
            Additional keyword arguments to be passed to the constructor of the module.

        Returns
        -------
        nn.Module
            The constructed neural network module.

        Notes
        -----
        This method is used to create a neural network module from a given
        configuration object. It is a class method, meaning it can be called
        on the class itself without the need for an instance.

        The `config` parameter should be an instance of the `modelConfig`
        class, which contains the necessary parameters for constructing the
        module. The `*args` and `**kwargs` parameters are used to pass
        additional arguments to the constructor of the module, if needed.

        As a special case for the Unet class:
        If the `config` object has a `context` attribute and the `cls` class
        has a `context_dim` parameter, the `context_dim` parameter of the
        config object will be set to the length of the `context` attribute.

        The method returns the constructed neural network module.

        Examples
        --------
        >>> config = modelConfig(...)
        >>> module = Unet.from_config(config, ...)
        """

        return config.construct(cls, *args, **kwargs)

    @classmethod
    def from_preset(cls: type, preset: str, *args: Any, **kwargs: Any) -> Self:
        """
        Construct a neural network module from a given preset.

        Parameters
        ----------
        cls : type
            The class of the neural network module.
        preset : str
            The name of the preset to use for constructing the module.
        *args : Any
            Additional positional arguments to be passed to the constructor of the module.
        **kwargs : Any
            Additional keyword arguments to be passed to the constructor of the module.

        Returns
        -------
        nn.Module
            The constructed neural network module.
        """
        config = modelConfig.from_preset(preset)
        return config.construct(cls, *args, **kwargs)

    @classmethod
    def load(
        cls: type,
        model_name: str,
        ckpt: Union[str, Path] = "best",
        model_parent: Union[Path, None] = None,
        logger: Union[logging.Logger, None] = None,
    ) -> Self:
        """
        Load a model from a checkpoint.

        Parameters
        ----------
        ckpt : str or Path
            The path to the checkpoint file.

        Returns
        -------
        nn.Module
            The loaded model.
        """
        ckpt = parse_lightning_ckpt(
            ckpt, model_name=model_name, model_parent=model_parent
        )
        (logger.info if logger is not None else print)(
            f"Loading model from checkpoint: {ckpt}"
        )
        return cls.load_from_checkpoint(ckpt, map_location="cpu")


class configModuleBase(nn.Module):
    """
    A base class for modules that use a configuration object.

    This class provides a class method to construct an instance of the class
    using a configuration object.
    """

    @classmethod
    def from_config(
        cls: type, config: modelConfig, *args: Any, **kwargs: Any
    ) -> nn.Module:
        """
        Construct a neural network module from a given configuration.

        Parameters
        ----------
        cls : type
            The class of the neural network module.
        config : modelConfig
            The configuration object containing the necessary parameters for constructing the module.
        *args : Any
            Additional positional arguments to be passed to the constructor of the module.
        **kwargs : Any
            Additional keyword arguments to be passed to the constructor of the module.

        Returns
        -------
        nn.Module
            The constructed neural network module.

        Notes
        -----
        This method is used to create a neural network module from a given
        configuration object. It is a class method, meaning it can be called
        on the class itself without the need for an instance.

        The `config` parameter should be an instance of the `modelConfig`
        class, which contains the necessary parameters for constructing the
        module. The `*args` and `**kwargs` parameters are used to pass
        additional arguments to the constructor of the module, if needed.

        As a special case for the Unet class:
        If the `config` object has a `context` attribute and the `cls` class
        has a `context_dim` parameter, the `context_dim` parameter of the
        config object will be set to the length of the `context` attribute.

        The method returns the constructed neural network module.

        Examples
        --------
        >>> config = modelConfig(...)
        >>> module = Unet.from_config(config, ...)
        """
        # Special case for Unet
        if (
            hasattr(config, "context")
            and "context_dim" in signature(cls).parameters.keys()
        ):
            # config.context_dim = len(config.context)
            pass
        return config.construct(cls, *args, **kwargs)


def clamp_tensor(x):
    """
    Clamp the tensor values to avoid overflow or underflow.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        Clamped tensor.
    """
    dtype = torch.half if torch.is_autocast_enabled() else x.dtype
    clamp_value = torch.finfo(dtype).max - 1000
    x = torch.clamp(x, min=-clamp_value, max=clamp_value)
    return x


def zero_module(module):
    """
    Sets all parameters of a module to zero. Used for initializing the
    optimizers.

    Parameters
    ----------
    module : nn.Module
        Module to be zeroed.

    Returns
    -------
    nn.Module
        Zeroed module.
    """
    for param in module.parameters():
        param.detach().zero_()
    return module


def upsample(in_channels, out_channels=None, use_conv=True):
    """
    Upsampling layer, NxN --> 2Nx2N, using nearest neighbor algorithm.
    Basically, every pixel is quadrupled, and the new pixels are filled with
    the value of the original pixel.

    Parameters
    ----------
    in_channels : int
        Input channels
    out_channels : int, optional
        Output channels, if None (default) the number of channels is conserved.
    use_conv : bool, optional
        Whether to apply convolution layer after upsampling. True by default.

    Returns
    -------
    nn.Sequential
        Upsampling layer.
    """
    return nn.Sequential(
        # Upsampling quadruples each pixel.
        nn.Upsample(scale_factor=2, mode="nearest"),
        # Convolution leaves image size unchanged.
        (
            nn.Conv2d(in_channels, (out_channels or in_channels), 3, padding=1)
            if use_conv
            else nn.Identity()
        ),
    )


def downsample(in_channels, out_channels=None):
    """
    Downsampling layer, NxN -> N/2 x N/2. Works by splitting image into 4, then
    doing 1x1 convolution using the 4 sub-images as input channels.
    In the original U-Net, this is done by max pooling.

    Parameters
    ----------
    in_channels : int
        Input channels
    out_channels : int, optional
        Output channels, if None (default) the number of channels is conserved.

    Returns
    -------
    nn.Sequential
        Downsampling layer.
    """
    return nn.Sequential(
        # Rearrange: Split each image into 4 smaller and concatenate along
        # channel dimension.
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        # Convolution leaves image sizes unchanged, changes channel dimensions
        # back to original (or specified dim_out).
        nn.Conv2d(in_channels * 4, (out_channels or in_channels), 1),
    )


class TimestepBlock(nn.Module):
    """
    Abstract base class for any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        emb : torch.Tensor
            Timestep embeddings.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    Sequential module where forward() takes timestep embeddings as a second
    argument.
    """

    def forward(self, x, emb):
        """
        Apply the sequential module to `x` given `emb` timestep embeddings.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        emb : torch.Tensor
            Timestep embeddings.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class SinusoidalEmbedding(nn.Module):
    """
    Takes input t of shape (batch_size, 1) corresponding to the time values of
    the noised images, and returns embedding of shape (batch_size, dim).
    For a good explanation of the embedding formula, see:
    https://kazemnejad.com/blog/transformer_architecture_positional_encoding/

    """

    def __init__(self, dim):
        """
        Initialize the SinusoidalEmbedding layer.

        Parameters
        ----------
        dim : int
            Number of dimensions of the embedding vector.
        """
        super().__init__()
        self.dim = dim

    def forward(self, time):
        """
        Forward pass of the SinusoidalEmbedding layer.

        Parameters
        ----------
        time : torch.Tensor
            Input tensor of shape (batch_size, 1) representing the time values.

        Returns
        -------
        torch.Tensor
            Embedding tensor of shape (batch_size, dim).
        """
        device = time.device
        half_dim = self.dim // 2
        freqs = math.log(1e5) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, device=device) * -freqs)
        embeddings = time[:, None] * freqs[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class FourierEmbedding(nn.Module):
    """
    Takes input t of shape (batch_size, 1) corresponding to values of a
    continuous context feature, and returns
    embedding of shape (batch_size, dim).
    For a good explanation of the embedding formula, see:
    https://bmild.github.io/fourfeat/
    """

    def __init__(self, dim, scale=16):
        """
        Initialize the FourierEmbedding layer.

        Parameters
        ----------
        dim : int
            The dimension of the layer.
        scale : int, optional
            Scaling factor for the frequencies, by default 16.
        """
        super().__init__()
        self.dim = dim
        self.register_buffer("freqs", torch.randn(dim // 2) * scale)

    def forward(self, x):
        """
        Forward pass of the FourierEmbedding layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Embedding tensor.
        """
        x = x.outer((2 * np.pi * self.freqs.to(x.device)).to(x.dtype))
        embeddings = torch.cat([x.cos(), x.sin()], dim=-1)
        return embeddings


class LinearFeatureEmbedding(nn.Module):
    """
    Linear embedding module, which is used to inject context information into the
    model.
    """

    def __init__(self, dim_in, dim_out):
        """
        Initialize the LinearFeatureEmbedding layer.

        Parameters
        ----------
        dim_in : int
            Input dimension.
        dim_out : int
            Output dimension.
        """
        super().__init__()
        self.lin1 = nn.Linear(dim_in, dim_out)
        self.act = nn.GELU()
        self.lin2 = nn.Linear(dim_out, dim_out)

    def forward(self, x):
        """
        Forward pass of the LinearFeatureEmbedding layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        x = self.lin1(x)
        x = self.act(x)
        x = self.lin2(x)
        x = self.act(x)
        return x


class AdditiveContextEmbedding(nn.Module):
    """
    Takes input t of shape (batch_size, context_dim) corresponding to the
    values of the entire context, and returns embedding
    of shape (batch_size, dim), which is the sum of embeddings
    of all single features. Works for different embedding layers.
    """

    def __init__(self, context_dim, emb_cls, **cls_kwargs):
        """
        Initialize the AdditiveContextEmbedding layer.

        Parameters
        ----------
        context_dim : int
            Dimension of the context.
        emb_cls : nn.Module
            Embedding layer class.
        **cls_kwargs : dict
            Additional keyword arguments to be passed to the embedding class.
        """
        super().__init__()
        self.emb_layers = [emb_cls(**cls_kwargs) for _ in range(context_dim)]
        for i, layer in enumerate(self.emb_layers):
            self.add_module(f"emb_{i}", layer)

    def forward(self, x):
        """
        Forward pass of the AdditiveContextEmbedding layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Embedding tensor.
        """
        print(f"Context on {x.device}")
        return sum(
            layer(x[:, i].view(-1, 1)) for i, layer in enumerate(self.emb_layers)
        )


class WeightStandardizedConv2d(nn.Conv2d):
    """
    Weight-standardized 2d convolutional layer, built from a standard
    conv2d layer. Works better with group normalization.
    https://arxiv.org/abs/1903.10520
    https://kushaj.medium.com/weight-standardization-a-new-normalization-in-town-54b2088ce355 #noqa
    """

    def forward(self, x):
        """
        Forward pass of the WeightStandardizedConv2d layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3  # Epsilon
        weight = self.weight

        # Tensors with mean and variance, same shape as weight tensors
        # for subsequent operations.
        mean = weight.mean(dim=(1, 2, 3), keepdim=True)
        var = weight.var(dim=(1, 2, 3), unbiased=False, keepdim=True)
        # mean = reduce(weight, "o ... -> o 1 1 1", "mean")  # o = outp. channels
        # var = reduce(weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        out = F.conv2d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

        # Clamp inf values to avoid Infs/NaNs
        if torch.isinf(out).any():
            out = clamp_tensor(out)

        return out


class ResidualLinearAttention(nn.Module):
    """
    Basically the same as regular multi-head attention, but this implementation
    is more efficient, (linear vs quadratic).
    To be exact, when using softmax this is not precisely mathematically
    equivalent, but a very good approximation.
    https://arxiv.org/abs/1812.01243

    Parameters
    ----------
    dim : int
        Dimension of the input.
    heads : int, optional
        Number of attention heads, by default 4.
    dim_head : int, optional
        Dimension of each attention head, by default 32.
    """

    def __init__(self, dim, heads=4, head_channels=32):
        """
        Initialize the ResidualLinearAttention layer.

        Parameters
        ----------
        dim : int
            Dimension of the input.
        heads : int, optional
            Number of attention heads, by default 4.
        head_channels : int, optional
            Number of channels of each attention head, by default 32.
        """
        super().__init__()
        self.scale = head_channels**-0.5
        self.heads = heads
        hidden_dim = head_channels * heads
        self.pre_norm = nn.GroupNorm(32, dim)
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=True)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1, bias=True), nn.GroupNorm(1, dim)
        )

    def forward(self, x):
        """
        Forward pass of the ResidualLinearAttention layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        res = x
        _, _, h, w = x.shape

        # qkv: Tuple of 3 tensors of shape [b, dim_head*heads, h, w]:
        x = self.pre_norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=1)

        # Reshape to three tensors of shape [b, heads, dim_head, h*w]:
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q * self.scale

        # Trick to prevent overflow in softmax
        q_shift = q - q.amax(dim=-2, keepdim=True).detach()
        k_shift = k - k.amax(dim=-1, keepdim=True).detach()

        q_norm = q_shift.softmax(dim=-2)
        k_norm = k_shift.softmax(dim=-1)

        context = torch.einsum("b h d n, b h e n -> b h d e", k_norm, v)
        out = torch.einsum("b h d e, b h d n -> b h e n", context, q_norm)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)

        return self.to_out(out) + res


class ResidualBlock(TimestepBlock):
    """
    Basic U-Net Block.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        emb_dim,
        *,
        dropout=0.0,
        norm_groups=32,
        up=False,
        down=False,
        use_conv=False,
    ):
        """
        Initialize the Layer class.

        Parameters
        ----------
        in_channels : int
            The input channels of the layer.
        out_channels : int
            The output channels of the layer.
        emb_dim : int
            The dimensions of the time embedding vector.
        dropout : float, optional
            The dropout probability, by default 0.0.
        norm_groups : int, optional
            The number of groups for group normalization, by default 32.
        up : bool, optional
            Whether to perform upsampling, by default False.
        down : bool, optional
            Whether to perform downsampling, by default False.
        use_conv : bool, optional
            Whether to use convolutional layers, for upsampling by default False.
            If true, a 3x3 convolution is also applied to the residual layer in
            the case of resampling. If false, this convolution is 1x1, i.e. FCN.
        """
        super().__init__()

        self.resize = up or down
        self.do_res_conv = in_channels != out_channels

        match norm_groups:
            case int() | float():
                norm_groups_in = norm_groups_out = norm_groups
            case (ng_in, ng_out):
                norm_groups_in = ng_in
                norm_groups_out = ng_out
            case _:
                raise ValueError(f"Invalid norm_groups: {norm_groups}")

        # Input layers
        self.in_layers = nn.Sequential(
            OrderedDict(
                [
                    ("group_norm", nn.GroupNorm(norm_groups_in, in_channels)),
                    ("silu", nn.SiLU()),
                    (
                        "conv",
                        WeightStandardizedConv2d(
                            in_channels, out_channels, 3, padding=1, bias=True
                        ),
                    ),
                ]
            )
        )

        # Resampling layers
        if up:
            self.h_upd = upsample(in_channels, use_conv=False)
            self.x_upd = upsample(in_channels, use_conv=False)

        elif down:
            self.h_upd = downsample(in_channels)
            self.x_upd = downsample(in_channels)

        # Time embedding layers
        self.time_emb_dim = emb_dim
        self.emb_layers = None
        if self.time_emb_dim:
            self.emb_layers = nn.Sequential(
                OrderedDict(
                    [
                        ("silu", nn.SiLU()),
                        ("linear", nn.Linear(emb_dim, out_channels * 2)),
                    ]
                )
            )

        # Output layers
        self.out_layers = nn.Sequential(
            OrderedDict(
                [
                    ("group_norm", nn.GroupNorm(norm_groups_out, out_channels)),
                    ("silu", nn.SiLU()),
                    ("dropout", nn.Dropout(p=dropout)),
                    (
                        "conv",
                        zero_module(
                            WeightStandardizedConv2d(
                                out_channels, out_channels, 3, padding=1, bias=True
                            )
                        ),
                    ),
                ]
            )
        )

        # Residual layers
        if self.do_res_conv:
            self.res_conv = (
                WeightStandardizedConv2d(in_channels, out_channels, 3)
                if use_conv
                else nn.Conv2d(in_channels, out_channels, 1)
            )

    def forward(self, x, time_emb=None):
        """
        Forward pass of the layer.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        time_emb : torch.Tensor
            The time embedding tensor.

        Returns
        -------
        torch.Tensor
            The output tensor after applying the forward pass.
        """
        # Input Layers
        if self.resize:
            # Split input layers into Norm+SiLU and Conv
            # (up/downsample will happen in between)
            in_pre, in_conv = self.in_layers[:-1], self.in_layers[-1]

            # Hidden state
            h = in_pre(x)  # Norm+SiLU
            h = self.h_upd(h)  # Up/downsample
            h = in_conv(h)  # Conv

            # Residual
            x = self.x_upd(x)  # Up/downsample residual

        else:
            h = self.in_layers(x)

        # Time embedding
        if time_emb is not None:
            assert (
                self.time_emb_dim is not None
            ), "Time embedding is not defined for this layer."
            time_emb = self.emb_layers(time_emb)  # SiLU and Linear
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale, shift = time_emb.chunk(2, dim=1)

        # Output layers
        # Split output layers into Norm and SiLU+Dropout+Conv
        # (time embedding will be applied in between)
        out_norm, out_post = self.out_layers[0], self.out_layers[1:]
        # Apply Norm and time embedding
        h = out_norm(h)
        if time_emb is not None:
            h = h * (scale + 1) + shift
        # Apply SiLU+Dropout+Conv
        h = out_post(h)
        # Residual layer
        if self.do_res_conv:
            x = self.res_conv(x)

        return clamp_tensor(h + x)


class ResidualBlockAttention(nn.Module):
    """
    Sequential module that applies a residual block and an attention block.
    """

    def __init__(self, resBlock, attnBlock):
        """
        Initialize the class.

        Parameters
        ----------
        resBlock : ResBlock
            The ResBlock object.
        attnBlock : AttnBlock
            The AttnBlock object.
        """
        super().__init__()
        self.resBlock = resBlock
        self.attnBlock = attnBlock

    def forward(self, x, time_emb=None):
        """
        Forward pass of the layer.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        time_emb : torch.Tensor
            The time embedding tensor.

        Returns
        -------
        torch.Tensor
            The output tensor after passing through the layer.
        """
        x = self.resBlock(x, time_emb)
        x = self.attnBlock(x)
        return x


class DownsampleBlock(ResidualBlock):
    """
    Basic U-Net block with downsampling.
    """

    def __init__(
        self,
        channels,
        emb_dim,
        *,
        dropout=0.0,
        norm_groups=32,
    ):
        """
        Initialize the Layer class.

        Parameters
        ----------
        channels : int
            The number of input and output channels.
        emb_dim : int
            The dimension of the time embedding.
        dropout : float, optional
            The dropout rate, by default 0.0.
        norm_groups : int, optional
            The number of groups to normalize the input channels, by default 32.
        """
        super().__init__(
            channels,
            channels,
            emb_dim,
            dropout=dropout,
            norm_groups=norm_groups,
            up=False,
            down=True,
            use_conv=False,
        )


class UpsampleBlock(ResidualBlock):
    """
    Basic U-Net block with upsampling.
    """

    def __init__(
        self,
        channels,
        emb_dim,
        *,
        dropout=0.0,
        norm_groups=32,
    ):
        """
        Initialize the Layer class.

        Parameters
        ----------
        channels : int
            The number of input and output channels.
        emb_dim : int
            The dimension of the time embedding.
        dropout : float, optional
            The dropout rate, by default 0.0.
        norm_groups : int, optional
            The number of groups to normalize the channels, by default 32.
        """
        super().__init__(
            channels,
            channels,
            emb_dim,
            dropout=dropout,
            norm_groups=norm_groups,
            up=True,
            down=False,
            use_conv=False,
        )


class FeatureMapToScalarSequence(nn.Module):
    def __init__(self, in_channels, hidden_dim, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

        # Project input feature map channels to hidden_dim
        self.input_proj = nn.Linear(in_channels, hidden_dim)

        # Learnable queries for each sequence position
        self.query_embed = nn.Parameter(torch.randn(seq_len, hidden_dim))

        # Multi-head cross-attention: queries attend to spatial tokens
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=4, batch_first=True
        )

        # Final projection to scalar (D=1)
        self.to_scalar = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape

        # Flatten spatial grid and project
        x = x.view(B, C, H * W).permute(0, 2, 1)  # (B, H*W, C)
        x = self.input_proj(x)  # (B, H*W, hidden_dim)

        # Repeat queries for each item in the batch
        queries = self.query_embed.unsqueeze(0).repeat(B, 1, 1)  # (B, L, hidden_dim)

        # Cross-attention: queries attend to spatial features
        attended, _ = self.cross_attn(queries, x, x)  # (B, L, hidden_dim)

        # Project each token to a scalar
        output = self.to_scalar(attended).squeeze(-1)  # (B, L)

        return output  # shape: (B, L)


class CatalogEmbedding(nn.Module):

    def __init__(self, emb_dim, f=2, channels=(4, 1)):

        super().__init__()

        n_blocks = np.log2(f)
        assert int(n_blocks) == n_blocks, f"{f=} must be a power of 2."
        n_blocks = int(n_blocks)

        match channels:

            case int() | float():
                ch_in = ch_out = channels
            case (ch_in, ch_out):
                ch_in, ch_out = channels
            case _:
                raise ValueError(f"Invalid channels: {channels}")

        self.blocks = nn.Sequential(
            OrderedDict(
                [
                    (
                        f"in_block_{i + 1}",
                        ResidualBlock(
                            ch_in,
                            ch_out if (i == n_blocks - 1) else ch_in,
                            emb_dim=None,
                            down=True,
                            norm_groups=(
                                (ch_in, ch_out) if (i == n_blocks - 1) else ch_in
                            ),  # Instance Norm
                        ),
                    )
                    for i in range(n_blocks)
                ]
            )
        )
        self.proj = FeatureMapToScalarSequence(
            in_channels=ch_out,
            hidden_dim=emb_dim,
            seq_len=emb_dim,
        )

    def forward(self, x):
        """
        Forward pass of the CatalogEmbedding layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Embedding tensor.
        """
        feature_map = self.blocks(x)
        emb = self.proj(feature_map)
        return feature_map, emb


def param_grad_nan_hook(grad):
    if grad is not None and not torch.isfinite(grad).all():
        raise RuntimeError("NaN/Inf gradient detected in in_proj_weight!")
    return grad


class CatalogTopKEncoder(nn.Module):
    def __init__(self, in_dim=5, model_dim=128, n_heads=4, n_layers=2):
        super().__init__()
        self.model_dim = model_dim
        self.proj = nn.Linear(in_dim, model_dim)
        self.pos_mlp = nn.Sequential(nn.Linear(2, model_dim), nn.SiLU())
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=n_heads,
            dim_feedforward=4 * model_dim,
            batch_first=True,
            layer_norm_eps=1e-4,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self._register_nan_hooks()

    def forward(self, cat):  # cat: (B, K, 4) with (Δx, Δy, logF, logR)
        pos = cat[..., :2]
        x = self.proj(cat)
        pos_emb = self.pos_mlp(pos)
        x = x + pos_emb

        x = self.encoder(x)  # (B, K, model_dim)

        # Stupid testing
        # x = x * torch.nan

        if torch.isnan(x).any():
            raise NaNDetectedError("NaN detected in CatalogTopKEncoder output.")

        return x

    def _register_nan_hooks(self):
        # --- Parameter-level hooks ---
        # Module hooks
        for module_name, module in self.named_modules():
            if module is self:
                continue
            for param_name, param in self.named_parameters(recurse=False):
                if param.requires_grad:

                    def make_param_hook(m_name, p_name):
                        def param_hook(grad):
                            if grad is None:
                                return
                            mask = ~torch.isfinite(grad)
                            if mask.any():
                                num_nan = mask.sum().item()
                                # Optional: print indices if small
                                idx_info = ""
                                if grad.numel() <= 1000:
                                    idx_info = (
                                        f", indices: {mask.nonzero(as_tuple=True)}"
                                    )
                                raise NaNDetectedError(
                                    f"NaN/Inf detected in gradient of parameter '{p_name}' "
                                    f"in module '{m_name}', shape={tuple(grad.shape)}, "
                                    f"num NaN/Inf={num_nan}{idx_info}"
                                )
                            return grad

                        return param_hook

                    param.register_hook(make_param_hook(module_name, param_name))


import torch
import torch.nn as nn


class ResidualCrossAttention(nn.Module):
    """
    Residual cross-attention block using torch.nn.MultiheadAttention.
    Compatible with diffusion U-Nets conditioned on catalog tokens.
    """

    def __init__(self, dim, context_dim=None, heads=8, dropout=0.0, f_downsample=1):
        super().__init__()
        context_dim = context_dim or dim

        self.norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,  # input shape (B, N, C)
        )

        # Linear projection for context (to match key/value dim)
        # self.context_proj = nn.Linear(context_dim, dim)
        # Edit: CNN projection
        if f_downsample > 1:
            assert (
                f_downsample % 2 == 0
            ), f"f_downsample must be a power of 2, got {f_downsample}."
            self.context_proj = nn.Sequential(
                # Rearrange: Split each image into 4 smaller and concatenate along
                # channel dimension.
                Rearrange(
                    "b c (h p1) (w p2) -> b (c p1 p2) h w",
                    p1=f_downsample,
                    p2=f_downsample,
                ),
                # Convolution leaves image sizes unchanged, changes channel dimensions
                # back to original (or specified dim_out).
                nn.Conv2d(context_dim * (f_downsample**2), dim, 3, padding=1),
            )
        else:
            self.context_proj = nn.Conv2d(
                in_channels=context_dim,
                out_channels=dim,
                kernel_size=3,
                padding=1,
            )

        # Optional output projection + dropout (built into MHA)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context, need_weights=False):
        """
        x:        [B, C, H, W]   — U-Net feature map
        context:  [B, K, context_dim] — catalog tokens
        """
        B, C, H, W = x.shape
        N = H * W

        # flatten spatial dims → (B, N, C)
        x_flat = x.flatten(2).transpose(1, 2)
        x_norm = self.norm(x_flat)

        # project context to same embedding dim as x
        context_proj = self.context_proj(context).flatten(2).transpose(1, 2)

        # MultiheadAttention: query=x, key/value=context
        attn_out, attn_weights = self.cross_attn(
            query=x_norm,
            key=context_proj,
            value=context_proj,
            need_weights=need_weights,
        )

        # residual connection
        out = x_flat + self.dropout(attn_out)
        # reshape back to (B, C, H, W)
        if need_weights:
            return out.transpose(1, 2).view(B, C, H, W), attn_weights
        else:
            return out.transpose(1, 2).view(B, C, H, W)


class AttentionGate(nn.Module):
    def __init__(self, g_channels, s_channels, out_channels):
        super().__init__()
        self.Wg = nn.Sequential(
            nn.Conv2d(g_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
        )
        self.Ws = nn.Sequential(
            nn.Conv2d(s_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(out_channels, 1, kernel_size=1), nn.Sigmoid()
        )

    def forward(self, g, s):
        g1 = self.Wg(g)  # Decoder features
        s1 = self.Ws(s)  # Skip connection features
        out = F.relu(g1 + s1)  # Merge signals
        psi = self.psi(out)  # Attention map (0 to 1)
        return s * psi  # Filtered skip
