from inspect import signature

import torch
import torch.nn as nn

from models.networks.modules import *


class Decoder(nn.Module):
    def __init__(
        self,
        init_channels,
        z_channels,
        image_channels,
        channel_mults=(
            1,
            2,
            4,
        ),
        norm_groups=32,
        dropout=0.0,
        num_res_blocks=2,
        attention_levels=0,
        attention_heads=4,
        attention_head_channels=32,
    ):
        super().__init__()

        # Determine channels
        self.input_channels = image_channels
        self.z_channels = z_channels
        self.init_channels = init_channels
        ch = self.init_channels * channel_mults[-1]
        n_levels = len(channel_mults)

        # Initial convolution layer
        self.init_conv = zero_module(
            WeightStandardizedConv2d(self.z_channels, ch, 3, padding=1)
        )

        # Create middle block
        self.middle_block = nn.Sequential(
            ResidualBlock(ch, ch, emb_dim=0, dropout=dropout, norm_groups=norm_groups),
            ResidualLinearAttention(
                ch, heads=attention_heads, head_channels=attention_head_channels
            ),
            ResidualBlock(ch, ch, emb_dim=0, dropout=dropout, norm_groups=norm_groups),
        )

        # Create list of upsampling blocks
        self.up_blocks = nn.ModuleList([])

        # Fill up-blocks (loop in reverse, hence [::-1])
        for res_level, mult in list(enumerate(channel_mults))[::-1]:
            for i in range(num_res_blocks + 1):

                # Create residual block
                block = ResidualBlock(
                    ch,
                    int(mult * init_channels),
                    emb_dim=0,
                    dropout=dropout,
                    norm_groups=norm_groups,
                )
                ch = int(mult * self.init_channels)

                # Add attention layer to block if necessary
                if n_levels - res_level <= attention_levels:
                    attn = ResidualLinearAttention(
                        ch, heads=attention_heads, head_channels=attention_head_channels
                    )
                    block = ResidualBlockAttention(block, attn)

                # Add residual(-attention) block to up-list
                self.up_blocks.append(block)

                # Add upsample block if not last resolution level
                if res_level and i == num_res_blocks:
                    block = UpsampleBlock(
                        ch, emb_dim=0, dropout=dropout, norm_groups=norm_groups
                    )
                    self.up_blocks.append(block)

        # Final residual block
        self.out = nn.Sequential(
            nn.GroupNorm(norm_groups, ch),
            nn.SiLU(),
            zero_module(nn.Conv2d(ch, self.input_channels, 3, padding=1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the decoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, z_channels, height/8, width/8).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, input_channels, height, width).
        """
        # Initial convolution
        x = self.init_conv(x)

        # Middle block
        x = self.middle_block(x)

        # Upsampling blocks
        for block in self.up_blocks:
            x = block(x)

        # Final output layer
        x = self.out(x)

        return x


class Encoder(nn.Module):

    def __init__(
        self,
        init_channels,
        z_channels,
        image_channels,
        channel_mults=(
            1,
            2,
            4,
        ),
        norm_groups=32,
        dropout=0.0,
        num_res_blocks=2,
        attention_levels=0,
        attention_heads=4,
        attention_head_channels=32,
        variational=True,  # Will affect latent space dimension (2 * z_channels if True, z_channels if False
    ):
        super().__init__()

        # Determine channels
        self.input_channels = image_channels
        self.z_channels = z_channels
        self.init_channels = init_channels

        # Initial convolution layer
        self.init_conv = WeightStandardizedConv2d(
            self.input_channels, self.init_channels, 3, padding=1
        )

        # Create list of downsampling blocks
        self.down_blocks = []
        ch = self.init_channels
        n_levels = len(channel_mults)

        # Fill down-block list
        for res_level, mult in enumerate(channel_mults):
            for _ in range(num_res_blocks):
                # Create residual block
                block = ResidualBlock(
                    ch,
                    int(mult * init_channels),
                    emb_dim=0,  # Embedding dimension = 0 --> Block won't have emb. layer
                    dropout=dropout,
                    norm_groups=norm_groups,
                )
                ch = int(mult * init_channels)

                # Add attention layer to block if necessary
                if n_levels - res_level <= attention_levels:
                    attn = ResidualLinearAttention(
                        ch, heads=attention_heads, head_channels=attention_head_channels
                    )
                    block = ResidualBlockAttention(block, attn)

                # Add residual(-attention) block to down-list
                self.down_blocks.append(block)

            # Add downsample block if not last resolution level
            if res_level != n_levels - 1:
                block = DownsampleBlock(
                    ch, emb_dim=0, dropout=dropout, norm_groups=norm_groups
                )
                self.down_blocks.append(block)

        # Make nn.Sequential for torch compilation
        self.down_blocks = nn.Sequential(*self.down_blocks)

        # Create middle block
        self.middle_block = nn.Sequential(
            ResidualBlock(ch, ch, emb_dim=0, dropout=dropout, norm_groups=norm_groups),
            ResidualLinearAttention(
                ch, heads=attention_heads, head_channels=attention_head_channels
            ),
            ResidualBlock(ch, ch, emb_dim=0, dropout=dropout, norm_groups=norm_groups),
        )

        # End block: Mapping to latent space
        self.end_block = nn.Sequential(
            nn.GroupNorm(norm_groups, ch),
            nn.SiLU(),
            WeightStandardizedConv2d(
                ch, (2 if variational else 1) * z_channels, 3, padding=1
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_channels, height, width).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, z_channels, height/8, width/8).
        """
        # Initial convolution
        x = self.init_conv(x)

        # Downsampling blocks
        x = self.down_blocks(x)

        # for block in self.down_blocks:
        #     x = block(x)

        # Middle block
        x = self.middle_block(x)

        # End block: Mapping to latent space
        x = self.end_block(x)

        return x
