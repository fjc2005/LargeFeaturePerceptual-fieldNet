import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from typing import Union, Tuple, List


class Conv(nn.Conv2d):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 dilation: int = 1,
                 groups: int = 1):
        K = kernel_size + (dilation - 1) * 2
        padding = (K - 1) // 2
        super().__init__(in_channels=in_channels, out_channels=out_channels,
                         kernel_size=kernel_size, padding=padding, stride=stride,
                         dilation=dilation, groups=groups)


class BPConv(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3):
        super().__init__()
        self.conv1 = Conv(in_channels, out_channels, kernel_size=1)
        self.conv2 = Conv(out_channels, out_channels, kernel_size=kernel_size, groups=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)

        return x


class DilationBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 hid_channels: int = None,
                 out_channels: int = None,
                 kernel_size: int = 3,
                 act: str = 'SiLU'):
        super().__init__()
        if hid_channels is None:
            hid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels
        self.conv_input = Conv(in_channels, hid_channels, kernel_size=1)
        self.conv_dil_1 = Conv(hid_channels, hid_channels, kernel_size=kernel_size, dilation=1, groups=hid_channels)
        self.conv_dil_2 = Conv(hid_channels, hid_channels, kernel_size=kernel_size, dilation=2, groups=hid_channels)
        self.conv_dil_3 = Conv(hid_channels, hid_channels, kernel_size=kernel_size, dilation=2, groups=hid_channels)
        self.conv_output = Conv(hid_channels, out_channels, kernel_size=1)
        assert act in ['SiLU', 'ReLU', 'GELU']
        if act == 'SiLU':
            self.act = nn.SiLU(inplace=True)
        elif act == 'ReLU':
            self.act = nn.ReLU(inplace=True)
        elif act == 'GELU':
            self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_input(x)
        residual = x
        x = self.conv_dil_1(x)
        x = self.act(x)
        x = self.conv_dil_2(x)
        x = self.act(x)
        x = self.conv_dil_3(x)
        x = self.act(x)
        x = x + residual
        x = self.conv_output(x)

        return x


class BPConvBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 hid_channels: int = None,
                 out_channels: int = None,
                 kernel_size: int = 3,
                 act: str = 'SiLU'):
        super().__init__()
        if hid_channels is None:
            hid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels
        self.conv_input = Conv(in_channels, hid_channels, kernel_size=1)
        self.conv_bp_1 = BPConv(hid_channels, hid_channels, kernel_size=kernel_size)
        self.conv_bp_2 = BPConv(hid_channels, hid_channels, kernel_size=kernel_size)
        self.conv_bp_3 = BPConv(hid_channels, hid_channels, kernel_size=kernel_size)
        self.conv_output = Conv(hid_channels, out_channels, kernel_size=1)
        assert act in ['SiLU', 'ReLU', 'GELU']
        if act == 'SiLU':
            self.act = nn.SiLU(inplace=True)
        elif act == 'ReLU':
            self.act = nn.ReLU(inplace=True)
        elif act == 'GELU':
            self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_input(x)
        residual = x
        x = self.conv_bp_1(x)
        x = self.act(x)
        x = self.conv_bp_2(x)
        x = self.act(x)
        x = self.conv_bp_3(x)
        x = self.act(x)
        x = x + residual
        x = self.conv_output(x)

        return x


class BasicConvBlock(nn.Module):
    def __init__(self,
                 d_model: int,
                 reduction: int = 2,
                 act: str = 'SiLU'):
        super().__init__()
        assert d_model % reduction == 0
        hid_channels = d_model // reduction
        self.DilConv = DilationBlock(in_channels=d_model,
                                     hid_channels=hid_channels,
                                     act=act)
        self.BPConv = BPConvBlock(in_channels=d_model,
                                  hid_channels=hid_channels,
                                  act=act)
        assert act in ['SiLU', 'ReLU', 'GELU']
        if act == 'SiLU':
            self.act = nn.SiLU(inplace=True)
        elif act == 'ReLU':
            self.act = nn.ReLU(inplace=True)
        elif act == 'GELU':
            self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        residual = self.DilConv(residual)
        x = self.BPConv(x)
        x = x + residual

        return x


class ESA(nn.Module):
    """
    Modification of Enhanced Spatial Attention (ESA), which is proposed by
    `Residual Feature Aggregation Network for Image Super-Resolution`
    Note: `conv_max` and `conv3_` are NOT used here, so the corresponding codes
    are deleted.
    """

    def __init__(self, esa_channels, n_feats, conv):
        super(ESA, self).__init__()
        f = esa_channels
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        c3 = self.conv3(v_max)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)),
                           mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)
        return x * m


class LFPBlock(nn.Module):
    def __init__(self,
                 d_model: int,
                 reduction: int = 2,
                 act: str = 'SiLU'):
        super().__init__()
        assert d_model % reduction == 0
        self.conv = BPConv(d_model, d_model, kernel_size=3)
        self.body = BasicConvBlock(d_model=d_model,
                                   reduction=reduction,
                                   act=act)
        self.esa = ESA(esa_channels=16,
                       n_feats=d_model,
                       conv=nn.Conv2d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.body(x)
        x = self.esa(x)
        x = x + residual
        x = self.conv(x)

        return x


class LFPGroup(nn.Module):
    def __init__(self,
                 n_blocks: int,
                 d_model: int,
                 reduction: int = 2,
                 act: str = 'SiLU'):
        super().__init__()
        blocks = []
        for i in range(n_blocks):
            blocks.append(LFPBlock(d_model=d_model,
                                   reduction=reduction,
                                   act=act))
        self.body = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.body(x)

        return x


class upsampler(nn.Module):
    def __init__(self,
                 d_model: int,
                 out_channels: int,
                 upscale: int = 2):
        super().__init__()
        self.conv = Conv(d_model, out_channels * upscale * upscale, kernel_size=3)
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.upsampler(x)

        return x


@ARCH_REGISTRY.register()
class LFPN(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 n_groups: int,
                 n_blocks: int,
                 d_model: int,
                 reduction: int = 2,
                 act: str = 'SiLU',
                 img_range: float = 255.0,
                 rgb_mean: Union[Tuple, List] = (0.4488, 0.4371, 0.4040),
                 upscale: int = 2):
        super().__init__()
        assert d_model % reduction == 0
        self.img_range = img_range
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        self.feature_extract = BPConv(in_channels, d_model, kernel_size=3)
        groups = []
        for i in range(n_groups):
            groups.append(LFPGroup(n_blocks=n_blocks,
                                   d_model=d_model,
                                   reduction=reduction,
                                   act=act))
        self.body = nn.Sequential(*groups)
        self.conv_res = nn.Conv2d(d_model * n_groups, d_model, kernel_size=1)
        self.conv_after_body = BPConv(d_model, d_model, kernel_size=3)
        self.upsampler = upsampler(d_model=d_model,
                                   out_channels=out_channels,
                                   upscale=upscale)
        assert act in ['SiLU', 'ReLU', 'GELU']
        if act == 'SiLU':
            self.act = nn.SiLU(inplace=True)
        elif act == 'ReLU':
            self.act = nn.ReLU(inplace=True)
        elif act == 'GELU':
            self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range
        x = self.feature_extract(x)
        # Long-Skip Connection
        res = x
        residual = []
        for group in self.body:
            x = group(x)
            residual.append(x)
        x = torch.cat(residual, dim=1)
        x = self.conv_res(x)
        x = self.act(x)
        x = self.conv_after_body(x)
        x = x + res
        # up-sample
        x = self.upsampler(x)
        x = x / self.img_range + self.mean

        return x
