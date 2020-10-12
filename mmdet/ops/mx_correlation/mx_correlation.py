import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.module import Module

from . import mx_correlation_cuda


class MxCorrelationFunction(Function):

    @staticmethod
    def forward(ctx,
                input1,
                input2,
                pad_size,
                kernel_size,
                max_displacement,
                stride1,
                stride2):
        ctx.pad_size = pad_size
        ctx.kernel_size = kernel_size
        ctx.max_displacement = max_displacement
        ctx.stride1 = stride1
        ctx.stride2 = stride2
        ctx.input1_size = input1.size()
        ctx.input2_size = input2.size()

        with torch.cuda.device_of(input1):
            rbot1 = input1.new()
            rbot2 = input2.new()
            output = input1.new()

            mx_correlation_cuda.forward(input1, input2, rbot1, rbot2, output,
                         pad_size, kernel_size, max_displacement,
                         stride1, stride2)
        ctx.save_for_backward(rbot1, rbot2)

        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        rbot1, rbot2 = ctx.saved_tensors

        with torch.cuda.device_of(rbot1):
            grad_input1 = rbot1.new_zeros(ctx.input1_size)
            grad_input2 = rbot2.new_zeros(ctx.input2_size)

            mx_correlation_cuda.backward(
                rbot1, rbot2,
                grad_output, grad_input1, grad_input2,
                ctx.pad_size, ctx.kernel_size, ctx.max_displacement,
                ctx.stride1, ctx.stride2)

        return grad_input1, grad_input2, None, None, None, None, None, None


class MxCorrelation(Module):
    def __init__(self,
                 pad_size=0,
                 kernel_size=0,
                 max_displacement=0,
                 stride1=1,
                 stride2=2):
        super(MxCorrelation, self).__init__()
        self.pad_size = pad_size
        self.kernel_size = kernel_size
        self.max_displacement = max_displacement
        self.stride1 = stride1
        self.stride2 = stride2

    def forward(self, input1, input2):

        result = MxCorrelationFunction.apply(
            input1, input2, self.pad_size, self.kernel_size, self.max_displacement,
            self.stride1, self.stride2)

        return result
