import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.module import Module

from . import psroi_pool_cuda


class PSRoIPoolingFunction(Function):

    @staticmethod
    def forward(ctx, features, rois, pooled_height, pooled_width,
            spatial_scale, group_size, output_dim):
        assert features.is_cuda
        assert isinstance(pooled_height, int) and isinstance(pooled_width, int)
        ctx.save_for_backward(rois)
        num_rois = rois.size()[0]
        out_size = (num_rois, output_dim, pooled_height, pooled_width)
        output = features.new_zeros(out_size)
        mappingchannel = features.new_zeros(out_size, dtype=torch.int)

        psroi_pool_cuda.forward(pooled_height, pooled_width, spatial_scale,
            group_size, output_dim, features, rois, output, mappingchannel);

        ctx.mappingchannel = mappingchannel
        ctx.rois = rois
        ctx.spatial_scale = spatial_scale
        ctx.output_dim = output_dim
        ctx.feature_size = features.size()

        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        assert grad_output.is_cuda
        assert(ctx.feature_size is not None and grad_output.is_cuda)
        spatial_scale = ctx.spatial_scale
        output_dim = ctx.output_dim
        feature_size = ctx.feature_size
        mappingchannel = ctx.mappingchannel
        rois = ctx.saved_tensors[0]
        assert feature_size is not None

        grad_input = grad_rois = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.new_zeros(feature_size)
            psroi_pool_cuda.backward(spatial_scale, output_dim,
                grad_output, rois, grad_input, mappingchannel)
        return grad_input, None, None, None, None, None, None


psroi_pool = PSRoIPoolingFunction.apply


class PSRoIPool(Module):
    def __init__(self, pooled_height, pooled_width, spatial_scale, group_size, output_dim):
        super(PSRoIPool, self).__init__()

        self.pooled_width = int(pooled_width)
        self.pooled_height = int(pooled_height)
        self.spatial_scale = float(spatial_scale)
        self.group_size = int(group_size)
        self.output_dim = int(output_dim)

    def forward(self, features, rois):
        return psroi_pool(features, rois,
                self.pooled_height, self.pooled_width, self.spatial_scale,
                self.group_size, self.output_dim)

    def __repr___(self):
        format_str = self.__class__.__name__
        format_str += '(out_size={}, spatial_scale={}, group_size={} output_dim={}'.format(
            (self.pooled_height, self.pooled_width), self.spatial_scale,
            self.group_size, self.output_dim)
        return format_str
