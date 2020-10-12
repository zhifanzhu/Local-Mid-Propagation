from __future__ import division

import torch
import torch.nn as nn

from mmdet import ops
from ..registry import ROI_EXTRACTORS


@ROI_EXTRACTORS.register_module
class MultiRoIExtractor(nn.Module):
    """Extract RoI features from multi levels feature map(as well as original image).

    If there are mulitple input feature levels, each RoI is mapped to a level
    according to its scale.

    Args:
        roi_layer (dict): Specify RoI layer type and arguments.
        out_channels (int): Output channels of RoI layers.
        featmap_strides (int): Strides of input feature maps.
        with_original_image (bool): Whether to add original image when ROIpooling.
        finest_scale (int): Scale threshold of mapping to level 0.
    """

    def __init__(self,
                 roi_layer,
                 out_channels,
                 featmap_strides,
                 # with_original_image = True,
                 finest_scale=56):
        super(MultiRoIExtractor, self).__init__()
        self.out_channels = out_channels
        self.featmap_strides = featmap_strides  ## [P2, P3, P4, P5]
        # self.with_original_image = with_original_image
        self.finest_scale = finest_scale
        # if self.with_original_image:
        #   self.featmap_strides.append(1) ## [P2, P3, P4, P5, Img]
        self.roi_layers = self.build_roi_layers(roi_layer, self.featmap_strides)

    @property
    def num_inputs(self):
        """int: Input feature map levels."""
        return len(self.featmap_strides)  ## 4

    def init_weights(self):
        pass

    def build_roi_layers(self, layer_cfg, featmap_strides):
        cfg = layer_cfg.copy()
        layer_type = cfg.pop('type')
        assert hasattr(ops, layer_type)
        layer_cls = getattr(ops, layer_type)
        roi_layers = nn.ModuleList(
            [layer_cls(spatial_scale=1 / s, **cfg) for s in featmap_strides])
        # if self.with_original_image:
        #   roi_layers.append(layer_cls(spatial_scale = 1, **cfg)) ## for original image
        return roi_layers

    def map_roi_levels(self, rois, num_levels):
        """Map rois to corresponding feature levels by scales.

        - scale < finest_scale: level 0
        - finest_scale <= scale < finest_scale * 2: level 1
        - finest_scale * 2 <= scale < finest_scale * 4: level 2
        - scale >= finest_scale * 4: level 3

        Args:
            rois (Tensor): Input RoIs, shape (k, 5), [batch_ind, x1, y1, x2, y2]
            num_levels (int): Total level number.

        Returns:
            Tensor: Level index (0-based) of each RoI, shape (k, )
        """
        scale = torch.sqrt(
            (rois[:, 3] - rois[:, 1] + 1) * (rois[:, 4] - rois[:, 2] + 1))
        target_lvls = torch.floor(torch.log2(scale / self.finest_scale + 1e-6))
        target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).long()
        return target_lvls

    ########## Updated By LCZ, time: 2019.1.24 ##########
    def forward(self, feats, rois):
        ## feats: [P2, P3, P4, P5]
        ## norm_img: normalized original image
        if len(feats) == 1:
            return self.roi_layers[0](feats[0], rois)

        out_size = self.roi_layers[0].out_size

        roi_feats = [torch.cuda.FloatTensor(rois.size()[0],
                                      self.out_channels, out_size, out_size).fill_(0)
                    for _ in range(len(feats))] ## 4
        # import pdb
        # pdb.set_trace()
        # feats = feats + (norm_img, )
        # roi_feats.append(torch.cuda.FloatTensor(rois.size()[0], 3, out_size, out_size).fill_(0)) ## add 3-d image, 4-->5

        # import pdb
        # pdb.set_trace()
        assert len(feats) == len(self.featmap_strides)
        num_levels = len(feats) ## 4 or 5

        for i in range(num_levels):
            roi_feats_t = self.roi_layers[i](feats[i], rois)
            roi_feats[i] += roi_feats_t

        ## roi_feats: list, features cropped from feature maps (and original image).
        ##              [P2, P3, P4, P5], P2~P5 are [K, 256, 7, 7]
        return roi_feats
