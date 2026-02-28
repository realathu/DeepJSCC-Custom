# -*- coding: utf-8 -*-
"""
Created on Tue Dec  11:00:00 2023

@author: chun
"""

import torch
import torch.nn as nn
from channel import Channel


""" def _image_normalization(norm_type):
    def _inner(tensor: torch.Tensor):
        if norm_type == 'nomalization':
            return tensor / 255.0
        elif norm_type == 'denormalization':
            return (tensor * 255.0).type(torch.FloatTensor)
        else:
            raise Exception('Unknown type of normalization')
    return _inner """


def ratio2filtersize(x: torch.Tensor, ratio: float) -> int:
    """Convert compression ratio to inner channel count.

    The encoder applies two stride-2 convolutions, so spatial dims are
    divided by 4 in each axis (factor 16 total area reduction).
    Computed analytically to avoid instantiating a dummy encoder model.
    """
    if x.dim() == 4:
        before_size = x[0].numel()
        h, w = x.shape[-2] // 4, x.shape[-1] // 4
    elif x.dim() == 3:
        before_size = x.numel()
        h, w = x.shape[-2] // 4, x.shape[-1] // 4
    else:
        raise Exception('Unknown size of input')
    c = before_size * ratio / (h * w)
    return max(1, int(c))


class _ConvWithPReLU(nn.Module):
    """Conv2d + optional BatchNorm + PReLU."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, use_bn=True):
        super(_ConvWithPReLU, self).__init__()
        # bias is redundant when BN is present (BN re-centres activations)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                              bias=not use_bn)
        self.bn   = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.prelu = nn.PReLU()
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='leaky_relu')

    def forward(self, x):
        return self.prelu(self.bn(self.conv(x)))


class _TransConvWithPReLU(nn.Module):
    """ConvTranspose2d + optional BatchNorm + configurable activation."""
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 activate=None, padding=0, output_padding=0, use_bn=True):
        super(_TransConvWithPReLU, self).__init__()
        if activate is None:
            activate = nn.PReLU()
        self.transconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding,
            bias=not use_bn)
        self.bn       = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.activate = activate
        # use isinstance so the check works correctly (== compares identity, not type)
        if isinstance(activate, nn.PReLU):
            nn.init.kaiming_normal_(self.transconv.weight, mode='fan_out',
                                    nonlinearity='leaky_relu')
        else:
            nn.init.xavier_normal_(self.transconv.weight)

    def forward(self, x):
        return self.activate(self.bn(self.transconv(x)))


class _Encoder(nn.Module):
    def __init__(self, c=1, is_temp=False, P=1):
        super(_Encoder, self).__init__()
        self.is_temp = is_temp
        self.conv1 = _ConvWithPReLU(in_channels=3,  out_channels=16, kernel_size=5, stride=2, padding=2)
        self.conv2 = _ConvWithPReLU(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2)
        # conv3 + conv4 form a residual block (both 32→32, same spatial size)
        self.conv3 = _ConvWithPReLU(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        self.conv4 = _ConvWithPReLU(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        # conv5: no BN — output feeds directly into power normalisation
        self.conv5 = _ConvWithPReLU(in_channels=32, out_channels=2*c, kernel_size=5, padding=2,
                                    use_bn=False)
        self.norm = self._normlizationLayer(P=P)

    @staticmethod
    def _normlizationLayer(P=1):
        def _inner(z_hat: torch.Tensor):
            if z_hat.dim() == 4:
                batch_size = z_hat.size()[0]
                k = torch.prod(torch.tensor(z_hat.size()[1:]))
            elif z_hat.dim() == 3:
                batch_size = 1
                k = torch.prod(torch.tensor(z_hat.size()))
            else:
                raise Exception('Unknown size of input')
            z_temp = z_hat.reshape(batch_size, 1, 1, -1)
            z_trans = z_hat.reshape(batch_size, 1, -1, 1)
            tensor = torch.sqrt(P * k) * z_hat / torch.sqrt((z_temp @ z_trans))
            if batch_size == 1:
                return tensor.squeeze(0)
            return tensor
        return _inner

    def forward(self, x):
        x   = self.conv1(x)
        x   = self.conv2(x)
        res = x                        # residual shortcut: skip conv3 + conv4
        x   = self.conv3(x)
        x   = self.conv4(x) + res      # add shortcut — same shape (32 ch, same HW)
        if not self.is_temp:
            x = self.conv5(x)
            x = self.norm(x)
        return x


class _Decoder(nn.Module):
    def __init__(self, c=1):
        super(_Decoder, self).__init__()
        self.tconv1 = _TransConvWithPReLU(
            in_channels=2*c, out_channels=32, kernel_size=5, stride=1, padding=2)
        # tconv2 + tconv3 form a residual block (both 32→32, same spatial size)
        self.tconv2 = _TransConvWithPReLU(
            in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.tconv3 = _TransConvWithPReLU(
            in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.tconv4 = _TransConvWithPReLU(
            in_channels=32, out_channels=16, kernel_size=5, stride=2, padding=2, output_padding=1)
        # tconv5: no BN — Sigmoid output layer, want clean [0,1] range
        self.tconv5 = _TransConvWithPReLU(
            in_channels=16, out_channels=3, kernel_size=5, stride=2, padding=2,
            output_padding=1, activate=nn.Sigmoid(), use_bn=False)

    def forward(self, x):
        x   = self.tconv1(x)
        res = x                        # residual shortcut: skip tconv2 + tconv3
        x   = self.tconv2(x)
        x   = self.tconv3(x) + res     # add shortcut — same shape (32 ch, same HW)
        x   = self.tconv4(x)
        x   = self.tconv5(x)
        return x


class DeepJSCC(nn.Module):
    def __init__(self, c, channel_type='AWGN', snr=None):
        super(DeepJSCC, self).__init__()
        self.encoder = _Encoder(c=c)
        if snr is not None:
            self.channel = Channel(channel_type, snr)
        self.decoder = _Decoder(c=c)
        self.criterion = nn.MSELoss(reduction='mean')
        # Try to import SSIM at construction time; fall back to pure MSE gracefully
        try:
            from pytorch_msssim import ssim as _ssim_fn
            self._ssim_fn   = _ssim_fn
            self._use_ssim  = True
        except ImportError:
            self._use_ssim  = False

    def forward(self, x):
        z = self.encoder(x)
        if hasattr(self, 'channel') and self.channel is not None:
            z = self.channel(z)
        x_hat = self.decoder(z)
        return x_hat

    def change_channel(self, channel_type='AWGN', snr=None):
        if snr is None:
            self.channel = None
        else:
            self.channel = Channel(channel_type, snr)

    def get_channel(self):
        if hasattr(self, 'channel') and self.channel is not None:
            return self.channel.get_channel()
        return None

    def loss(self, prd, gt, alpha: float = 0.85):
        """Combined SSIM + MSE loss (α·SSIM_loss + (1-α)·MSE).

        Falls back to pure MSE if pytorch-msssim is not installed.
        prd / gt are expected in the [0, 255] range (after denorm).
        """
        mse = self.criterion(prd, gt)
        if self._use_ssim:
            # SSIM loss: 1 - SSIM so lower = better, matches MSE direction
            ssim_loss = 1.0 - self._ssim_fn(
                prd, gt, data_range=255.0, size_average=True)
            return alpha * ssim_loss + (1.0 - alpha) * mse
        return mse


if __name__ == '__main__':
    model = DeepJSCC(c=20)
    print(model)
    x = torch.rand(1, 3, 128, 128)
    y = model(x)
    print(y.size())
    print(y)
    print(model.encoder.norm)
    print(model.encoder.norm(y))
    print(model.encoder.norm(y).size())
    print(model.encoder.norm(y).size()[1:])
