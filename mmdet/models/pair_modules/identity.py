import torch.nn as nn

from ..registry import PAIR_MODULE

"""
Simplified from TEMPORAL_MODULE.
This file is a demonstration.
"""


@PAIR_MODULE.register_module
class Identity(nn.Module):
    """ Identity temporal module, i.e. no modification on input data.
    """

    def __init__(self):
        super(Identity, self).__init__()
        self.decoder = nn.Sequential()  # Identity Module

    def init_weights(self):
        pass

    def forward(self, feat, feat_ref, is_train=False):
        return feat
