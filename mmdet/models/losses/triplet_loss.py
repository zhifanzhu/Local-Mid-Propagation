import torch.nn as nn
import torch.nn.functional as F

from ..registry import LOSSES


@LOSSES.register_module
class TripletMarginLoss(nn.Module):

    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, anchors, postives, negatives):
        loss = self.loss_weight * F.triplet_margin_loss(
            anchors,
            postives,
            negatives)
        return loss
