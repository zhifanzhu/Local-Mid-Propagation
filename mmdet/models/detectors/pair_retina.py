from ..registry import DETECTORS
from .pair_single_stage import PairSingleStageDetector


@DETECTORS.register_module
class PairRetinaNet(PairSingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 pair_module=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(PairRetinaNet, self).__init__(
            backbone, neck, bbox_head, pair_module,
            train_cfg, test_cfg, pretrained)
