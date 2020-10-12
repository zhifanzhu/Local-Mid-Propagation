from mmcv.parallel import DataContainer as DC

from .pipelines import Compose
from .registry import DATASETS
from .imagenet_pairdet30 import PairDET30Dataset


@DATASETS.register_module
class TripleDET30Dataset(PairDET30Dataset):
    """ The pipeline is composed of a main pipeline and a `twin` pipeline. """

    CLASSES = ('n02691156', 'n02419796', 'n02131653', 'n02834778', 'n01503061', 'n02924116',
               'n02958343', 'n02402425', 'n02084071', 'n02121808', 'n02503517', 'n02118333',
               'n02510455', 'n02342885', 'n02374451', 'n02129165', 'n01674464', 'n02484322',
               'n03790512', 'n02324045', 'n02509815', 'n02411705', 'n01726692', 'n02355227',
               'n02129604', 'n04468005', 'n01662784', 'n04530566', 'n02062744', 'n02391049',)
    DATASET_NAME = 'vid'

    def __init__(self,
                 ann_file,
                 pipeline,
                 twin_pipeline,
                 middle='B',  # 'B' for big, 'S' for small
                 **kwargs):
        super(TripleDET30Dataset, self).__init__(
            ann_file, pipeline, **kwargs)
        self.twin_pipeline = Compose(twin_pipeline)
        self.middle = middle

    def prepare_train_img(self, idx):
        """ Pipelines:
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True, skip_img_without_anno=False),
            dict(type='Resize', img_scale=(512, 512), keep_ratio=False),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
        """
        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        results = self.pipeline(results)

        ref_results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(ref_results)
        big_results = self.twin_pipeline(ref_results)
        results['first_img'] = DC(big_results['img'].data.clone(), stack=True)
        if self.middle == 'B':
            results['second_img'] = DC(big_results['img'].data.clone(), stack=True)
        elif self.middle == 'S':
            results['second_img'] = DC(results['img'].data.clone(), stack=True)

        if len(results['gt_bboxes'].data) == 0:
            return None
        return results


