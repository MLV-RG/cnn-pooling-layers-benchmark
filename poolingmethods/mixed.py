import torch
import torch.nn as nn
import torch.nn.functional as F


class MixedPool2d(nn.Module):
    def __init__(self, kernel_size, stride):
        super(MixedPool2d, self).__init__()
        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda'
        self.max = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
        self.avg = nn.AvgPool2d(kernel_size=kernel_size, stride=stride)
        alpha = torch.Tensor(1).to(device)
        self.alpha = nn.Parameter(alpha, requires_grad=True)

        nn.init.uniform_(self.alpha, a=0.25, b=0.75)

    def forward(self, x):
        mix = self.alpha * self.max(x) + (1 - self.alpha) * self.avg(x)
        return mix
