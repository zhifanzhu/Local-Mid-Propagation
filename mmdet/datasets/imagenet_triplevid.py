import os.path as osp
import numpy as np

from .pipelines import Compose
from .registry import DATASETS
from .imagenet_pairvid import PairVIDDataset


@DATASETS.register_module
class TripleVIDDataset(PairVIDDataset):
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
                 middle='B',  # 'S'
                 preserve_order=True,
                 test_sampling_style='train',  # or 'key'
                 key_interval=10,
                 **kwargs):
        super(TripleVIDDataset, self).__init__(
            ann_file, pipeline, **kwargs)
        self.test_sampling_style = test_sampling_style
        self.key_interval = key_interval
        self.middle = middle
        self.preserve_order = preserve_order

        self.twin_pipeline = Compose(twin_pipeline)

    def __getitem__(self, idx):
        if self.test_mode:
            if self.test_sampling_style == 'train':
                raise NotImplementedError
                # return self.prepare_test_img_train(idx)
            elif self.test_sampling_style == 'key':
                return self.prepare_test_img_key(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

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
        frame_ind = img_info['frame_ind']
        num_frames = img_info['num_frames']
        foldername = img_info['foldername']
        offsets = np.random.choice(
            self.max_offset - self.min_offset + 1, 3, replace=False)
        offsets = offsets + frame_ind
        offsets = np.maximum(np.minimum(offsets, num_frames - 1), 0)
        if self.preserve_order:
            offsets = sorted(offsets)
        first, second, third = offsets

        ann_info = self.get_ann_info(idx, first)
        filename = osp.join(foldername, f"{first:06d}.JPEG")
        img_info['filename'] = filename
        first_results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(first_results)
        first_results = self.twin_pipeline(first_results)
        flip = first_results['img_meta'].data['flip']

        second_anno_info = self.get_ann_info(idx, second)
        second_filename = osp.join(foldername, f"{second:06d}.JPEG")
        img_info['filename'] = second_filename
        second_results = dict(img_info=img_info, ann_info=second_anno_info)
        self.pre_pipeline(second_results)
        if self.match_flip:
            second_results['flip'] = flip
        if self.middle == 'B':
            second_results = self.twin_pipeline(second_results)
        elif self.middle == 'S':
            second_results = self.pipeline(second_results)
        else:
            raise ValueError

        third_anno_info = self.get_ann_info(idx, third)
        third_filename = osp.join(foldername, f"{third:06d}.JPEG")
        img_info['filename'] = third_filename
        third_results = dict(img_info=img_info, ann_info=third_anno_info)
        self.pre_pipeline(third_results)
        if self.match_flip:
            third_results['flip'] = flip
        third_results = self.pipeline(third_results)

        results = third_results
        results['first_img'] = first_results['img']
        results['second_img'] = second_results['img']

        if len(results['gt_bboxes'].data) == 0:
            return None
        return results

    def prepare_test_img_key(self, idx):
        """
        dict(type='Collect', keys=['img'],
             meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape',
                'scale_factor', 'flip', 'img_norm_cfg', 'is_key')),
        """
        img_info = self.img_infos[idx]
        frame_ind = img_info['frame_ind']
        is_key = ((frame_ind % self.key_interval) == 0)
        foldername = img_info['foldername']
        ann_info = self.get_ann_info(idx, frame_ind)
        filename = osp.join(foldername, f"{frame_ind:06d}.JPEG")
        img_info['filename'] = filename
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        results['is_key'] = is_key
        results['frame_ind'] = frame_ind
        if is_key:
            results = self.twin_pipeline(results)
        else:
            results = self.pipeline(results)
        return results
