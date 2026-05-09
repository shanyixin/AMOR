"""GLCNet architecture for anatomy-aware shoulder landmark localization."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class LocalConvBlock(nn.Module):
    """Local convolution block used by the encoder, decoder, and prediction head."""

    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class AxialSSMBranch(nn.Module):
    """Multi-directional SSM aggregation branch used inside Res-M."""

    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.pre = nn.Sequential(conv1x1(dim, dim), nn.BatchNorm3d(dim), nn.ReLU(inplace=True))
        self.post = nn.Sequential(conv1x1(dim, dim), nn.BatchNorm3d(dim), nn.ReLU(inplace=True))
        self.ssm = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x):
        batch, channels = x.shape[:2]
        residual = self.pre(x)
        image_dims = residual.shape[2:]
        token_count = residual.shape[2:].numel()
        tokens = residual.reshape(batch, channels, token_count).transpose(-1, -2)

        directional_tokens = (
            self.ssm(tokens),
            torch.flip(self.ssm(torch.flip(tokens, dims=[2])), dims=[2]),
            torch.flip(self.ssm(torch.flip(tokens, dims=[1])), dims=[1]),
            torch.flip(self.ssm(torch.flip(tokens, dims=[1, 2])), dims=[1, 2]),
        )
        aggregated = torch.stack(directional_tokens, dim=0).mean(dim=0)
        grid = aggregated.transpose(-1, -2).reshape(batch, channels, *image_dims)
        return self.post(grid + residual)


class ResMBlock(nn.Module):
    """Residual convolution block augmented with the paper's Res-M SSM branch."""

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, ssm_branch=None, ssm_scale=1.0):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.ssm_branch = ssm_branch
        self.ssm_scale = ssm_scale
        self.downsample = downsample

    def forward(self, x):
        identity = x
        local = self.relu(self.bn1(self.conv1(x)))
        local = self.bn2(self.conv2(local))

        if self.ssm_branch is not None:
            local = local + self.ssm_scale * self.ssm_branch(x)
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(local + identity)


def make_res_m_stage(inplanes, planes, blocks, stride=1, ssm_branch=None):
    downsample = nn.Sequential(conv1x1(inplanes, planes, stride), nn.BatchNorm3d(planes))
    layers = [ResMBlock(inplanes, planes, stride, downsample)]
    for _ in range(1, blocks):
        layers.append(ResMBlock(planes, planes, ssm_branch=ssm_branch))
    return nn.Sequential(*layers)


class SkipAttention(nn.Module):
    """Lightweight stream recalibration for MDR skip fusion."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.gate(x).view(x.shape[0], x.shape[1], 1, 1, 1)


class MDRBlock(nn.Module):
    """Multi-dimensional Dynamic Recalibration decoder block."""

    def __init__(self, decoder_channels, encoder_channels, out_channels, reduction=16):
        super().__init__()
        self.decoder_gate = SkipAttention(decoder_channels, reduction)
        self.encoder_gate = SkipAttention(encoder_channels, reduction)
        self.local_fusion = LocalConvBlock(decoder_channels + encoder_channels, out_channels)

        hidden = max(1, out_channels // reduction)
        self.channel_mlp = nn.Sequential(
            nn.Linear(out_channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_channels, bias=False),
        )
        self.channel_sigmoid = nn.Sigmoid()
        self.spatial_gate = nn.Sequential(nn.Conv3d(2, 1, kernel_size=7, padding=3, bias=False), nn.Sigmoid())

    def forward(self, decoder_feature, encoder_feature):
        fused = self.local_fusion(torch.cat([self.decoder_gate(decoder_feature), self.encoder_gate(encoder_feature)], dim=1))
        batch, channels = fused.shape[:2]

        avg_desc = F.adaptive_avg_pool3d(fused, 1).view(batch, channels)
        max_desc = F.adaptive_max_pool3d(fused, 1).view(batch, channels)
        channel_gate = self.channel_sigmoid(self.channel_mlp(avg_desc) + self.channel_mlp(max_desc))
        channel_refined = fused * channel_gate.view(batch, channels, 1, 1, 1)

        avg_map = torch.mean(channel_refined, dim=1, keepdim=True)
        max_map, _ = torch.max(channel_refined, dim=1, keepdim=True)
        spatial_gate = self.spatial_gate(torch.cat([avg_map, max_map], dim=1))
        return channel_refined * spatial_gate


class CoordinateAwareAttentionHead(nn.Module):
    """CATH: axis-aware feature modulation followed by heatmap projection."""

    def __init__(self, in_channels, num_landmarks, reduction=32):
        super().__init__()
        hidden = max(8, in_channels // reduction)
        self.embedding = LocalConvBlock(in_channels, in_channels)
        self.shared = nn.Sequential(
            nn.Conv3d(in_channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm3d(hidden),
            nn.Hardswish(),
        )
        self.att_x = nn.Conv3d(hidden, in_channels, kernel_size=1)
        self.att_y = nn.Conv3d(hidden, in_channels, kernel_size=1)
        self.att_z = nn.Conv3d(hidden, in_channels, kernel_size=1)
        self.output = nn.Sequential(LocalConvBlock(in_channels, in_channels), nn.Conv3d(in_channels, num_landmarks, 1))

    def forward(self, x):
        decoded = self.embedding(x)
        context = self.shared(decoded)
        ax = torch.sigmoid(self.att_x(context.mean(dim=(3, 4), keepdim=True)))
        ay = torch.sigmoid(self.att_y(context.mean(dim=(2, 4), keepdim=True)))
        az = torch.sigmoid(self.att_z(context.mean(dim=(2, 3), keepdim=True)))
        return torch.sigmoid(self.output(decoded * ax * ay * az))


class GLCNet(nn.Module):
    """Global-to-Local Coordinate-aware Network described in the AMOR paper."""

    def __init__(self, in_ch=1, channels=32, blocks=3, number_classes=10):
        super().__init__()
        self.in_conv = LocalConvBlock(in_ch, channels, stride=2)

        self.layer1 = make_res_m_stage(channels, channels * 2, blocks, stride=2, ssm_branch=AxialSSMBranch(channels * 2))
        self.layer2 = make_res_m_stage(channels * 2, channels * 4, blocks, stride=2, ssm_branch=AxialSSMBranch(channels * 4))
        self.layer3 = make_res_m_stage(channels * 4, channels * 8, blocks, stride=2, ssm_branch=AxialSSMBranch(channels * 8))

        self.up5 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.mdr5 = MDRBlock(channels * 8, channels * 4, channels * 4)
        self.up6 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.mdr6 = MDRBlock(channels * 4, channels * 2, channels * 2)
        self.up7 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.mdr7 = MDRBlock(channels * 2, channels, channels)

        self.cath = CoordinateAwareAttentionHead(channels, number_classes)

    def forward(self, x):
        c1 = self.in_conv(x)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)

        c5 = self.mdr5(self.up5(c4), c3)
        c6 = self.mdr6(self.up6(c5), c2)
        c7 = self.mdr7(self.up7(c6), c1)
        heatmaps = self.cath(c7)
        return F.interpolate(heatmaps, size=x.shape[2:], mode="trilinear", align_corners=False)


nnMambaSeg = GLCNet
