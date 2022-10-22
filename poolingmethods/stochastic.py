import torch
import torch.nn as nn
import torch.nn.functional as F


class StochasticPool2d(nn.Module):

    def __init__(self, kernel_size=2, stride=2, padding=0):
        super(StochasticPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        torch.set_printoptions(threshold=10_000)
        device = 'cpu'
        if x.is_cuda:
            device = 'cuda'
        # batch size, channels, width, height
        batch, channels, w, h = x.data.shape  # [1,3,32,32], [100,6,28,28], [100,16,24,24]
        newshape_w = (w - self.kernel_size) // self.stride + 1
        newshape_h = (h - self.kernel_size) // self.stride + 1
        x = x.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)

        x = x.reshape(-1, self.kernel_size*self.kernel_size)

        # Multinomial does not play well with zero, so replace zeroes with ones.
        # That might favor the selection in the multinomial distributon but will not fail
        y = torch.ones(x.size()).to(device)
        nonzero_x = torch.where(x > 0, x, y).to(device)
        # Pick a random index based on multinomial distribution
        ids = torch.multinomial(nonzero_x, 1)
        x = x.gather(1, ids)
        x = x.reshape(batch, channels, newshape_w, newshape_h)
        # mask = torch.zeros(x.size()).to(device)
        # x = x * mask
        # x = x.reshape(-1, 4)
        # Select a random index for each window
        # idx = torch.randint(0, x.shape[1], size=(x.shape[0],)).type(torch.cuda.LongTensor).to(device)
        # x = x.contiguous().view(-1)
        # x = x.take(idx)
        # x = x.contiguous().view(b, c, w, h)
        # print(x)
        # print(x.shape)  # [1, 3, 16, 16, 2, 2]
        # x = x.reshape(b, c, -1, self.kernel_size * self.kernel_size)
        # print(x.shape)  # [1, 3, 256, 4]
        # x = x.permute(0, 1, 3, 2)
        # print(x.shape)  # [1, 3, 4, 256]
        # x = x.reshape(b, c * self.kernel_size * self.kernel_size, -1)
        # print(x.shape)  # [1, 12, 256]
        # x = F.fold(x, output_size=(w, h), kernel_size=self.kernel_size, stride=self.stride)
        # print(x.shape)  # [1, 3, 32, 32]
        # breakpoint()
        return x
