import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import weight_reduce_loss
from ..registry import LOSSES


@LOSSES.register_module
class BootstrappedSigmoidClassificationLoss(nn.Module):
    """From Google's Object Detction API:
    Bootstrapped sigmoid cross entropy classification loss function.

    This loss uses a convex combination of training labels and the current model's
    predictions as training targets in the classification loss. The idea is that
    as the model improves over time, its predictions can be trusted more and we
    can use these predictions to mitigate the damage of noisy/incorrect labels,
    because incorrect labels are likely to be eventually highly inconsistent with
    other stimuli predicted to have the same label by the model.

    In "soft" bootstrapping, we use all predicted class probabilities, whereas in
    "hard" bootstrapping, we use the single class favored by the model.

    See also Training Deep Neural Networks On Noisy Labels with Bootstrapping by
    Reed et al. (ICLR 2015).
    """

    def __init__(self,
                 alpha,
                 bootstrap_type='soft',
                 reduction='mean',
                 loss_weight=1.0):
        super(BootstrappedSigmoidClassificationLoss, self).__init__()
        assert bootstrap_type in ('soft', 'hard')
        self.alpha = alpha
        self.bootstrap_type = bootstrap_type
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        num_classes = cls_score.size(-1)
        target_tensor = F.one_hot(label, num_classes=num_classes).type(cls_score.dtype)
        if self.bootstrap_type == 'soft':
            bootstrap_target_tensor = self.alpha * target_tensor + (
                1.0 - self.alpha) * torch.sigmoid(cls_score)
        else:
            bootstrap_target_tensor = self.alpha * target_tensor + (
                1.0 - self.alpha) * (torch.sigmoid(cls_score) > 0.5).type(cls_score.dtype)

        per_entry_cross_ent = F.cross_entropy(cls_score,
                                              target=bootstrap_target_tensor.argmax(1),
                                              reduction=reduction)
        if weight is not None:
            weight = weight.float()
        loss = weight_reduce_loss(per_entry_cross_ent, weight,
                                  reduction=reduction, avg_factor=avg_factor)
        return loss
