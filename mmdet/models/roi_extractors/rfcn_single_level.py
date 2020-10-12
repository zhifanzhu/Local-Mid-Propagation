from __future__ import division

import torch.nn as nn
from mmcv.cnn import normal_init

from mmdet.ops import PSRoIPool
from mmdet.core import force_fp32
from ..registry import ROI_EXTRACTORS


@ROI_EXTRACTORS.register_module
class RfcnPSRoIExtractor(nn.Module):
    """Extract RoI features from a single level feature map.

    If there are mulitple input feature levels, each RoI is mapped to a level
    according to its scale.

    Args:
        finest_scale (int): Scale threshold of mapping to level 0.
    """

    def __init__(self,
                 featmap_stride,
                 num_classes,
                 reg_class_agnostic=False,
                 pooled_size=7,
                 ):
        super(RfcnPSRoIExtractor, self).__init__()
        num_reg_cls = 4 if reg_class_agnostic else 4 * num_classes
        self.cls_psroi_pool = PSRoIPool(
            pooled_size, pooled_size, 1.0 / featmap_stride, pooled_size, num_classes)
        self.loc_psroi_pool = PSRoIPool(
            pooled_size, pooled_size, 1.0 / featmap_stride, pooled_size, num_reg_cls)
        self.featmap_stride = featmap_stride
        self.fp16_enabled = False
        self.ps_cls_conv = nn.Conv2d(
            in_channels=512,
            out_channels=pooled_size*pooled_size*num_classes,
            kernel_size=1)
        self.ps_loc_conv = nn.Conv2d(
            in_channels=512,
            out_channels=pooled_size*pooled_size*num_reg_cls,
            kernel_size=1)
        self.featmap_strides = [self.featmap_stride]  # For test mixins

    @property
    def num_inputs(self):
        """int: Input feature map levels."""
        return 1

    def init_weights(self):
        normal_init(self.ps_cls_conv, mean=0.0, std=0.01, bias=0.0)
        normal_init(self.ps_loc_conv, mean=0.0, std=0.01, bias=0.0)

    @force_fp32(apply_to=('feats', ), out_fp16=True)
    def forward(self, feats, rois):
        assert len(feats) == 1
        cls_feat = self.ps_cls_conv(feats[0])
        loc_feat = self.ps_loc_conv(feats[0])
        cls_roi_feat = self.cls_psroi_pool(cls_feat, rois)
        loc_roi_feat = self.loc_psroi_pool(loc_feat, rois)
        return cls_roi_feat, loc_roi_feat
