"""Small U-Net sized for 32x32 GOES-19 patches and CPU-only training."""

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TinyUNet(nn.Module):
    """Encoder/decoder with `depth` downsampling stages.

    At the defaults (in_channels=16, base_channels=16, depth=3) this gives the
    channel plan 16->16->32->64->128 and, for a 32x32 input, a 4x4 bottleneck -
    a deeper encoder would shrink past the point of having a meaningful receptive
    field left. ~0.5M parameters, sized for CPU training.
    """

    def __init__(self, in_channels: int, base_channels: int, depth: int, out_channels: int = 1) -> None:
        super().__init__()
        channels = [in_channels] + [base_channels * (2**i) for i in range(depth + 1)]

        self.encoders = nn.ModuleList(ConvBlock(channels[i], channels[i + 1]) for i in range(depth))
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(channels[depth], channels[depth + 1])

        # Bilinear upsampling + conv instead of transposed conv - avoids checkerboard
        # artifacts and adds no extra learnable upsampling parameters.
        self.upsamples = nn.ModuleList(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False) for _ in range(depth)
        )
        self.decoders = nn.ModuleList(
            ConvBlock(channels[depth + 1 - i] + channels[depth - i], channels[depth - i]) for i in range(depth)
        )
        self.final = nn.Conv2d(channels[1], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for upsample, decoder, skip in zip(self.upsamples, self.decoders, reversed(skips)):
            x = upsample(x)
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)

        return self.final(x)
