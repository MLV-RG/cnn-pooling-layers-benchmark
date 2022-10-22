import torch.nn as nn
from torch.autograd import Function
import softpool_cuda
import torch.nn.functional as F
import torch
from torch.nn.modules.utils import _triple, _pair, _single


class CUDA_SOFTPOOL2d(Function):
    @staticmethod
    def forward(ctx, input, kernel=2, stride=None):
        # Create contiguous tensor (if tensor is not contiguous)
        no_batch = False
        if len(input.size()) == 3:
            no_batch = True
            input.unsqueeze_(0)
        B, C, H, W = input.size()
        kernel = _pair(kernel)
        if stride is None:
            stride = kernel
        else:
            stride = _pair(stride)
        oH = (H - kernel[0]) // stride[0] + 1
        oW = (W - kernel[1]) // stride[1] + 1
        output = input.new_zeros((B, C, oH, oW))
        softpool_cuda.forward_2d(input.contiguous(), kernel, stride, output)
        ctx.save_for_backward(input)
        ctx.kernel = kernel
        ctx.stride = stride
        if no_batch:
            return output.squeeze_(0)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Create contiguous tensor (if tensor is not contiguous)
        grad_input = torch.zeros_like(ctx.saved_tensors[0])
        saved = [grad_output.contiguous()] + list(ctx.saved_tensors) + [ctx.kernel, ctx.stride] + [grad_input]
        softpool_cuda.backward_2d(*saved)
        # Gradient underflow
        saved[-1][torch.isnan(saved[-1])] = 0
        return saved[-1], None, None


class SoftPool(nn.Module):

    def __init__(self, kernel_size=2, stride=2, force_inplace=False):
        super(SoftPool, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.force_inplace = force_inplace

    def forward(self, x):
        if x.is_cuda and not self.force_inplace:
            x = CUDA_SOFTPOOL2d.apply(x, self.kernel_size, self.stride)
            # Replace `NaN's if found
            if torch.isnan(x).any():
                return torch.nan_to_num(x)
            return x
        kernel_size = _pair(self.kernel_size)
        if self.stride is None:
            stride = kernel_size
        else:
            stride = _pair(self.stride)
        # Get input sizes
        _, c, h, w = x.size()
        # Create exponential mask (should be similar to max-like pooling)
        e_x = torch.sum(torch.exp(x), dim=1, keepdim=True)
        e_x = torch.clamp(e_x, float(0), float('inf'))
        # Apply mask to input and pool and calculate the exponential sum
        # Tensor: [b x c x d] -> [b x c x d']
        x = F.avg_pool2d(x.mul(e_x), kernel_size, stride=stride).mul_(sum(kernel_size)).div_(
            F.avg_pool2d(e_x, kernel_size, stride=stride).mul_(sum(kernel_size)))
        return torch.clamp(x, float(0), float('inf'))
