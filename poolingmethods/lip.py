import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Lip2d(nn.Module):

    def __init__(self, logit, kernel_size=2, stride=2, padding=0):
        super(Lip2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.logit = logit
        self.padding = padding

    def forward(self, x):
        device = 'cpu'
        if x.is_cuda:
            device = 'cuda'
        weight = self.logit.exp().to(device)
        return F.avg_pool2d(x * weight, self.kernel_size, self.stride, self.padding) / F.avg_pool2d(weight,
                                                                                                    self.kernel_size,
                                                                                                    self.stride,
                                                                                                    self.padding)
