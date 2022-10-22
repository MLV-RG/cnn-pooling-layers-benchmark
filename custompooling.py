from collections import OrderedDict

import torch.nn as nn
from torch import Tensor
import torch
from poolingmethods.spatial import SpatialPyramidPooling
from poolingmethods.softpool import SoftPool
from poolingmethods.spectral import SpectralPool2d
from poolingmethods.stochastic import StochasticPool2d
from poolingmethods.fuzzy import FuzzyPool2d
from poolingmethods.wavelet.util.wavelet_pool2d import StaticWaveletPool2d
from poolingmethods.mixed import MixedPool2d
from poolingmethods.gated import GatedPool2d
from poolingmethods.tree import Tree_level2
from poolingmethods.lip import Lip2d
import pywt
from torch.functional import F


class SoftGate(nn.Module):
    def __init__(self):
        super(SoftGate, self).__init__()

    def forward(self, x):
        return torch.sigmoid(x).mul(12.0).to('cuda')


class CustomPooling(nn.Module):
    def __init__(self, pooling_function='max'):
        super(CustomPooling, self).__init__()
        self.pooling_type = pooling_function

    def forward(self, x: Tensor) -> Tensor:
        if len(x.size()) < 4:
            x = x.view(1, x.size(0), x.size(1), x.size(2))
        batch, channels, w, h = x.data.shape

        if self.pooling_type == 'average':
            pooling_function = nn.AvgPool2d(kernel_size=2, stride=2)
        elif self.pooling_type == 'fractional':
            pooling_function = nn.FractionalMaxPool2d(2, output_ratio=(0.5, 0.5))
        elif self.pooling_type == 'stochastic':
            pooling_function = StochasticPool2d(kernel_size=2, stride=2)
        elif self.pooling_type == 'fuzzy':
            pooling_function = FuzzyPool2d(kernel_size=2, stride=2)
        elif self.pooling_type == 'l2':
            pooling_function = nn.LPPool2d(norm_type=2, kernel_size=2, stride=2)
        elif self.pooling_type == 'adaptive_max':
            pooling_function = nn.AdaptiveMaxPool2d((int(x.size(2) / 2), int(x.size(3) / 2)))
        elif self.pooling_type == 'spp':
            pooling_function = SpatialPyramidPooling()
        elif self.pooling_type == 'softpool':
            pooling_function = SoftPool(kernel_size=2, stride=2)
        elif self.pooling_type == 'wavelet':
            pooling_function = StaticWaveletPool2d(wavelet=pywt.Wavelet('haar'),
                                                   use_scale_weights=False,
                                                   scales=3)
        elif self.pooling_type == 'mixed':
            pooling_function = MixedPool2d(kernel_size=2, stride=2)
        elif self.pooling_type == 'overlapping':
            pooling_function = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        elif self.pooling_type == 'gated':
            pooling_function = GatedPool2d(kernel_size=2, stride=2)
        elif self.pooling_type == 'tree':
            pooling_function = Tree_level2(kernel_size=2, stride=2)
        elif self.pooling_type == 'spectral':
            pooling_function = SpectralPool2d(scale_factor=0.5)
        elif self.pooling_type == 'strided_convolution':
            weights = nn.Parameter(torch.Tensor(channels).to('cuda'))
            pooling_function = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=2, stride=2,
                                         padding=0, bias=False).to('cuda')
        elif self.pooling_type == 'lip':

            logit = nn.Sequential(
                OrderedDict((
                    ('conv', nn.Conv2d(channels, channels, 3, padding=1, bias=False)),
                    ('bn', nn.InstanceNorm2d(channels, affine=True)),
                    ('gate', SoftGate()),
                ))
            ).to('cuda')
            pooling_function = Lip2d(logit(x), kernel_size=2, stride=2)
        else:
            pooling_function = nn.MaxPool2d(kernel_size=2, stride=2)
        output = pooling_function(x)
        return output
