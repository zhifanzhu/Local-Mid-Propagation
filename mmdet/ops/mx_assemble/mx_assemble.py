import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.module import Module

from . import mx_assemble_cuda


class MxAssembleFunction(Function):

    @staticmethod
    def forward(ctx,
                aff,
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
        ctx.aff_size = aff.size()
        ctx.input2_size = input2.size()

        with torch.cuda.device_of(aff):
            rbot2 = input2.new()
            output = input2.new_zeros(input2.size())

            mx_assemble_cuda.forward(aff, input2, rbot2, output,
                         pad_size, kernel_size, max_displacement,
                         stride1, stride2)
        ctx.save_for_backward(aff, rbot2)

        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        aff, rbot2 = ctx.saved_tensors

        with torch.cuda.device_of(grad_output):
            rgrad_output = grad_output.new()
            grad_aff = grad_output.new_zeros(ctx.aff_size)
            grad_input2 = grad_output.new_zeros(ctx.input2_size)

            mx_assemble_cuda.backward(
                grad_output, rgrad_output, rbot2,
                aff, grad_aff, grad_input2,
                ctx.pad_size, ctx.kernel_size, ctx.max_displacement,
                ctx.stride1, ctx.stride2)

        return grad_aff, grad_input2, None, None, None, None, None, None


class MxAssemble(Module):
    def __init__(self, k):
        super(MxAssemble, self).__init__()
        self.pad_size = k
        self.kernel_size = 1
        self.max_displacement = k
        self.stride1 = 1
        self.stride2 = 1

    def forward(self, aff, input2):

        result = MxAssembleFunction.apply(
            aff, input2, self.pad_size, self.kernel_size, self.max_displacement,
            self.stride1, self.stride2)

        return result
