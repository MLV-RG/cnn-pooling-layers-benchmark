import torch
import torch.nn as nn
import torch.nn.functional as F


class Tree_level2(nn.Module):
    """Custom layer for Tree based pooling of tree level-2
    Arguments:
        kernel_size   : Single integer denoting the dimension of the square kernel
        stride        : Single integer denoting the equal stride in both directions
    Returns:
        output        : Tree pooling of depth 2 of input layer
    """

    def __init__(self, kernel_size, stride):
        super(Tree_level2, self).__init__()
        self.pool = kernel_size
        self.s = stride

        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda'

        v1 = torch.Tensor(1, 1, self.pool, self.pool).to(device)
        v2 = torch.Tensor(1, 1, self.pool, self.pool).to(device)
        w3 = torch.Tensor(1, 1, self.pool, self.pool).to(device)
        self.v1 = nn.Parameter(v1, requires_grad=True)
        self.v2 = nn.Parameter(v2, requires_grad=True)
        self.w3 = nn.Parameter(w3, requires_grad=True)

        nn.init.uniform_(self.v1, a=0.25, b=0.75)
        nn.init.uniform_(self.v2, a=0.25, b=0.75)
        nn.init.uniform_(self.w3, a=0.25, b=0.75)

    def forward(self, x):
        self.batch, self.channels, self.row, self.column = x.shape[0], x.shape[1], x.shape[2], x.shape[3]

        leaf1 = F.conv2d(input=x.view(self.batch * self.channels, 1, self.row, self.column), weight=self.v1, bias=None,
                         stride=(self.s, self.s)).view(self.batch, self.channels, self.row // 2, self.column // 2)
        leaf2 = F.conv2d(input=x.view(self.batch * self.channels, 1, self.row, self.column), weight=self.v2, bias=None,
                         stride=(self.s, self.s)).view(self.batch, self.channels, self.row // 2, self.column // 2)
        root = F.conv2d(input=x.view(self.batch * self.channels, 1, self.row, self.column), weight=self.w3, bias=None,
                        stride=(self.s, self.s)).view(self.batch, self.channels, self.row // 2, self.column // 2)

        torch.sigmoid_(root)
        output = torch.add(torch.mul(leaf1, root), torch.mul(leaf2, (1 - root)))
        return output
