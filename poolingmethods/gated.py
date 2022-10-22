import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedPool2d(nn.Module):
    """Custom Layer for Gated pooling, with one gate for all windows across all dimensions
        Arguments:
            kernel_size   : Single integer denoting the dimension of the square kernel
            stride        : Single integer denoting the equal stride in both directions
        Returns:
            gated         : Gated pooling between max and avg pooling, with a single gate for each window
        """

    def __init__(self, kernel_size, stride):
        super(GatedPool2d, self).__init__()
        self.pool = kernel_size
        self.stride = stride
        self.max = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
        self.avg = nn.AvgPool2d(kernel_size=kernel_size, stride=stride)
        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda'
        mask = torch.Tensor(1, 1, self.pool, self.pool).to(device)
        self.mask = nn.Parameter(mask, requires_grad=True)

        nn.init.normal_(self.mask, mean=0.0, std=1.0)

    def forward(self, x):
        self.batch, self.channels, self.row, self.column = x.shape[0], x.shape[1], x.shape[2], x.shape[3]

        # Changing the dimensions from (batch, channels, row, column) to (batch*channels, 1, row, column)
        # Allows us to use the standard convolution operation with the weight being same across all the channels, ie,
        # Using an esentially 2-D weight instead of a three dimensional one

        z = F.conv2d(input=x.view(self.batch * self.channels, 1, self.row, self.column), weight=self.mask, bias=None,
                     stride=(self.stride, self.stride)).view(self.batch, self.channels, self.row // 2, self.column // 2)
        z = torch.sigmoid(z)
        gated = torch.add(torch.mul(self.max(x), z), torch.mul(self.avg(x), (1 - z)))
        return gated
