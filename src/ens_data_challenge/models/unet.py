import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=True
        )
        self.norm = nn.InstanceNorm2d(out_channels, eps=1e-5, affine=True)
        self.activation = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        return self.activation(self.norm(self.conv(x)))


class EncoderStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, stride=stride)
        self.conv2 = ConvBlock(out_channels, out_channels, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DecoderStage(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channels, in_channels, kernel_size=2, stride=2, bias=False)
        self.conv1 = ConvBlock(
            in_channels + skip_channels, out_channels, stride=1)
        self.conv2 = ConvBlock(out_channels, out_channels, stride=1)

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class PlainConvUNet(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 54, deepsupervision: bool = True):
        super().__init__()

        self.deepsupervision = deepsupervision
        features = [32, 64, 128, 256, 512, 512, 512]
        strides = [1, 2, 2, 2, 2, 2, 2]

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        # Encoder
        current_in_channels = in_channels
        for i in range(len(features)):
            self.encoder.append(
                EncoderStage(
                    in_channels=current_in_channels,
                    out_channels=features[i],
                    stride=strides[i]
                )
            )
            current_in_channels = features[i]

        # Decoder
        for i in range(len(features) - 1, 0, -1):
            self.decoder.append(
                DecoderStage(
                    in_channels=features[i],
                    skip_channels=features[i-1],
                    out_channels=features[i-1]
                )
            )

        # Deep Supervision Heads
        # features[0] = 32 (Resolution 256x256)
        # features[1] = 64 (Resolution 128x128)
        # features[2] = 128 (Resolution 64x64)
        self.heads = nn.ModuleList([
            nn.Conv2d(features[0], num_classes,
                      kernel_size=1, stride=1, bias=True),
            nn.Conv2d(features[1], num_classes,
                      kernel_size=1, stride=1, bias=True),
            nn.Conv2d(features[2], num_classes,
                      kernel_size=1, stride=1, bias=True)
        ])

    def forward(self, x):
        skips = []

        # Through the encoder
        for i, stage in enumerate(self.encoder):
            x = stage(x)
            if i < len(self.encoder) - 1:
                skips.append(x)

        # Through the decoder
        ds_outputs = []
        num_decoder_stages = len(self.decoder)

        for i, stage in enumerate(self.decoder):
            skip_idx = - (i + 1)
            x = stage(x, skips[skip_idx])

            if self.deepsupervision:
                if i == num_decoder_stages - 3:
                    ds_outputs.append(self.heads[2](x))
                elif i == num_decoder_stages - 2:
                    ds_outputs.append(self.heads[1](x))
                elif i == num_decoder_stages - 1:
                    ds_outputs.append(self.heads[0](x))

        if self.deepsupervision and self.training:
            return ds_outputs[::-1]
        else:
            if self.deepsupervision:
                return ds_outputs[-1]
            else:
                return self.heads[0](x)


class ParametricUNet(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 54, deepsupervision: bool = True):
        super().__init__()

        features: list = [32, 64, 128, 256]
        strides: list = [1, 2, 2, 2]

        self.deepsupervision = deepsupervision
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        current_in_channels = in_channels
        for i in range(len(features)):
            self.encoder.append(
                EncoderStage(
                    in_channels=current_in_channels,
                    out_channels=features[i],
                    stride=strides[i]
                )
            )
            current_in_channels = features[i]

        for i in range(len(features) - 1, 0, -1):
            self.decoder.append(
                DecoderStage(
                    in_channels=features[i],
                    skip_channels=features[i-1],
                    out_channels=features[i-1]
                )
            )

        self.heads = nn.ModuleList()
        num_heads = min(3, len(features) - 1)
        for i in range(num_heads):
            self.heads.append(
                nn.Conv2d(features[i], num_classes,
                          kernel_size=1, stride=1, bias=True)
            )

    def forward(self, x):
        skips = []
        for i, stage in enumerate(self.encoder):
            x = stage(x)
            if i < len(self.encoder) - 1:
                skips.append(x)

        ds_outputs = []
        num_decoder_stages = len(self.decoder)

        for i, stage in enumerate(self.decoder):
            skip_idx = - (i + 1)
            x = stage(x, skips[skip_idx])

            if self.deepsupervision:
                num_heads = len(self.heads)
                if i >= num_decoder_stages - num_heads:
                    head_idx = num_decoder_stages - 1 - i
                    ds_outputs.append(self.heads[head_idx](x))

        if self.deepsupervision and self.training:
            return ds_outputs[::-1]
        else:
            if self.deepsupervision:
                return ds_outputs[-1]
            else:
                return self.heads[0](x)
