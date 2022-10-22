import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FuzzyPool2d(nn.Module):

    def __init__(self, kernel_size=2, stride=2):
        super(FuzzyPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        device = 'cpu'
        if x.is_cuda:
            device = 'cuda'
        # batch size, channels, width, height
        batch, channels, w, h = x.data.shape
        newshape_w = (w - self.kernel_size) // self.stride + 1
        newshape_h = (h - self.kernel_size) // self.stride + 1

        # rmax is set to 6, as defined in the original paper
        rmax = 6

        # Step 1, get the patches
        x = x.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        x = x.reshape(-1, self.kernel_size * self.kernel_size)

        # Step 2, calculate π for the triangular relationships of the paper

        m1 = x.clone()
        m2 = x.clone()
        m3 = x.clone()

        d = rmax / 2
        c = d / 3
        m1 = torch.where(m1 > d, 0., torch.where(m1 < c, 1., m1.mul(-1.).add(d).div(d - c).double()))

        a = rmax / 4
        m = rmax / 2
        b = m + a
        m2 = torch.where((m2 > a) | (m2 > b), 0.,
                         torch.where((m2 > b) & (m2 > m), m2.mul(-1.).add(b).div(b - m).double(),
                                     m2.sub(a).div(m - a).double()))

        r = rmax / 2
        q = r + rmax / 4
        m3 = torch.where(m1 > r, 0., torch.where(m1 > q, 1., m1.sub(r).div(q - r).double()))

        ms = torch.cat((m1, m2, m3), 1).reshape(x.size(0), 3, self.kernel_size * self.kernel_size).float()

        # Step 3, calculate sπ for all patches
        spi = ms.sum(2)

        # Step 4, calculate π' by getting argmax sπ and keeping that π
        argmaxs = spi.argmax(dim=1)
        pi_accented = ms.gather(1, argmaxs.view(-1, 1, 1).repeat(1, 1, ms.size(2))).reshape(-1, ms.size(2))

        # Let's calculate p'
        pi_accented_sums = pi_accented.sum(1)
        pi_accented_sums[pi_accented_sums == 0] = 1 # avoid division by zero

        pi_accented_patch_inner = pi_accented.reshape(x.size(0), self.kernel_size, self.kernel_size) @ x.reshape(
            x.size(0), self.kernel_size, self.kernel_size)
        pi_accented_patch_inner_sum = pi_accented_patch_inner.reshape(x.size(0),
                                                                      self.kernel_size * self.kernel_size).sum(1)

        # Now get the final patches and reconstruct the image
        x = pi_accented_patch_inner_sum.div(pi_accented_sums).reshape(batch, channels, newshape_w, newshape_h)

        return x
