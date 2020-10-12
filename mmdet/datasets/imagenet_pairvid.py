import os.path as osp
import xml.etree.ElementTree as ET
from pathlib import Path

import time
import mmcv
import numpy as np
from torch.utils.data import Dataset

from .pipelines import Compose
from .registry import DATASETS


@DATASETS.register_module
class PairVIDDataset(Dataset):

    CLASSES = ('n02691156', 'n02419796', 'n02131653', 'n02834778', 'n01503061', 'n02924116',
               'n02958343', 'n02402425', 'n02084071', 'n02121808', 'n02503517', 'n02118333',
               'n02510455', 'n02342885', 'n02374451', 'n02129165', 'n01674464', 'n02484322',
               'n03790512', 'n02324045', 'n02509815', 'n02411705', 'n01726692', 'n02355227',
               'n02129604', 'n04468005', 'n01662784', 'n04530566', 'n02062744', 'n02391049',)
    DATASET_NAME = 'vid'

    def __init__(self,
                 ann_file,
                 pipeline,
                 match_flip=False,
                 min_offset=-9,
                 max_offset=9,
                 min_size=None,
                 data_root=None,
                 img_prefix=None,
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False):
        self.ann_file = ann_file
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.seg_prefix = seg_prefix
        self.proposal_file = proposal_file
        self.test_mode = test_mode
        self.min_offset = min_offset
        self.max_offset = max_offset
        self.match_flip = match_flip

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.ann_file):
                self.ann_file = osp.join(self.data_root, self.ann_file)
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = osp.join(self.data_root, self.img_prefix)
            if not (self.seg_prefix is None or osp.isabs(self.seg_prefix)):
                self.seg_prefix = osp.join(self.data_root, self.seg_prefix)
            if not (self.proposal_file is None
                    or osp.isabs(self.proposal_file)):
                self.proposal_file = osp.join(self.data_root,
                                              self.proposal_file)
        # load annotations (and proposals)
        begin_time = time.time()
        self.img_infos = self.load_annotations(self.ann_file)
        print('load_annotations time: {:.1f}s from {}'
              .format(time.time() - begin_time, ann_file))
        if proposal_file is not None:
            self.proposals = self.load_proposals(proposal_file)
        else:
            self.proposals = None
        # filter images with no annotation during training
        if not test_mode:
            valid_inds = self._filter_imgs()
            self.img_infos = [self.img_infos[i] for i in valid_inds]
            if self.proposals is not None:
                self.proposals = [self.proposals[i] for i in valid_inds]
        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()
        # processing pipeline
        self.pipeline = Compose(pipeline)

        self.cat2label = {cat: i + 1 for i, cat in enumerate(self.CLASSES)}
        self.min_size = min_size

    def __len__(self):
        return len(self.img_infos)

    def load_annotations(self, ann_file):
        """
        training file would be: VID_train_15frames.txt
            train/ILSVRC2015_VID_train_0000/ILSVRC2015_train_00000000 1 10 300
            train/ILSVRC2015_VID_train_0000/ILSVRC2015_train_00000000 1 30 300

        val file would be: VID_val_videos.txt
            val/ILSVRC2015_val_00000000 1 0 464
            val/ILSVRC2015_val_00000001 465 0 464

        For train:
            one img_info is one frame in a video
        For val:
            one img_info is a whole video.
        """
        img_infos = []
        img_ids = mmcv.list_from_file(ann_file)

        def _train_get_vid_id(_id_line):
            _4d_8d, _1, _frame_ind, _num_frames = _id_line.split(' ')
            _frame_ind = int(_frame_ind)
            _num_frames = int(_num_frames)
            return _4d_8d, _frame_ind, _num_frames

        def _val_get_vid_id(_id_line):
            _vid_id, _cum, _0, _num_frames = _id_line.split(' ')
            return _vid_id, 0, int(_num_frames)

        is_eval = False
        if img_ids[0].split('/')[0] == 'train':
            vid_id_func = _train_get_vid_id
        elif img_ids[0].split('/')[0] == 'val':
            vid_id_func = _val_get_vid_id
            is_eval = True
        else:
            raise ValueError("Unknown prefix in annoation txt file.")

        for id_line in img_ids:
            # Probe first frame to get info
            video, frame_ind, num_frames = vid_id_func(id_line)
            foldername = f'Data/VID/{video}'
            xml_path = Path(self.img_prefix
                            )/f'Annotations/VID/{video}/{frame_ind:06d}.xml'
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            if is_eval:
                for frame_ind in range(0, num_frames):
                    img_infos.append(
                        dict(video=video,
                             foldername=foldername,
                             height=height,
                             width=width,
                             frame_ind=frame_ind,
                             num_frames=num_frames))
            else:
                img_infos.append(
                    dict(video=video,
                         foldername=foldername,
                         height=height,
                         width=width,
                         frame_ind=frame_ind,
                         num_frames=num_frames))

        return img_infos

    def get_ann_info(self, idx, frame_id):
        """
        Although img_info['frame_ind'] contains frame_ind for 'idx', we still need a
        parameter frame_id because frame_id might be different from that in 'img_info',
        e.g.  reference frame 's frame_ind
        """
        video = self.img_infos[idx]['video']
        xml_path = Path(self.img_prefix
                        )/f'Annotations/VID/{video}/{frame_id:06d}.xml'
        tree = ET.parse(xml_path)
        root = tree.getroot()
        bboxes = []
        labels = []
        trackids = []
        bboxes_ignore = []
        labels_ignore = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            # Track offset by 1, as we leave 0 for negatives
            trackid = int(obj.find('trackid').text) + 1
            label = self.cat2label[name]
            bnd_box = obj.find('bndbox')
            bbox = [
                int(bnd_box.find('xmin').text),
                int(bnd_box.find('ymin').text),
                int(bnd_box.find('xmax').text),
                int(bnd_box.find('ymax').text)
            ]
            ignore = False
            if self.min_size:
                assert not self.test_mode
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                if w < self.min_size or h < self.min_size:
                    ignore = True
            if ignore:
                bboxes_ignore.append(bbox)
                labels_ignore.append(label)
            else:
                bboxes.append(bbox)
                labels.append(label)
                trackids.append(trackid)
        if not bboxes:
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0, ))
            trackids = np.zeros((0, ))
        else:
            bboxes = np.array(bboxes, ndmin=2) - 1
            labels = np.array(labels)
            trackids = np.array(trackids)
        if not bboxes_ignore:
            bboxes_ignore = np.zeros((0, 4))
            labels_ignore = np.zeros((0, ))
        else:
            bboxes_ignore = np.array(bboxes_ignore, ndmin=2) - 1
            labels_ignore = np.array(labels_ignore)
        ann = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            trackids=trackids.astype(np.int64),
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            labels_ignore=labels_ignore.astype(np.int64))
        return ann

    def pre_pipeline(self, results):
        results['img_prefix'] = self.img_prefix
        results['seg_prefix'] = self.seg_prefix
        results['proposal_file'] = self.proposal_file
        results['bbox_fields'] = []
        results['mask_fields'] = []

    def _filter_imgs(self, min_size=32):
        """Filter images too small."""
        valid_inds = []
        for i, img_info in enumerate(self.img_infos):
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.img_infos[i]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1

    def _rand_another(self, idx):
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
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
        foldername = img_info['foldername']
        num_frames = img_info['num_frames']
        ann_info = self.get_ann_info(idx, frame_ind)
        filename = osp.join(foldername, f"{frame_ind:06d}.JPEG")
        img_info['filename'] = filename
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        results = self.pipeline(results)
        flip = results['img_meta'].data['flip']

        ref_frame_ind = max(
            min(frame_ind + np.random.randint(self.min_offset, self.max_offset+1),
                num_frames - 1), 0)
        ref_ann_info = self.get_ann_info(idx, ref_frame_ind)
        ref_filename = osp.join(foldername, f"{ref_frame_ind:06d}.JPEG")
        img_info['filename'] = ref_filename
        ref_results = dict(img_info=img_info, ann_info=ref_ann_info)

        self.pre_pipeline(ref_results)
        if self.match_flip:
            # Matched flip training cause poor results,
            # This is unknown...
            ref_results['flip'] = flip
        ref_results = self.pipeline(ref_results)

        results['ref_img'] = ref_results['img']

        if len(results['gt_bboxes'].data) == 0:
            return None
        return results

    def prepare_test_img(self, idx):
        """
        For first frame, provide is_first==True, and identical img_cache
        for non-first frame, provide is_first==False, and previous img as img_cache

        Test time, no data shuffle. As this happens in MSRA & MaskTrackRCNN
        Currently, we don't use keyframe mode.

        Dataloader does not pass 'img_prev' to model, it's up to model to choose
        to store 'img_prev', or intermediate result. Thus no side-effect happens.

        Testpipeline:
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(512, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=False),
                    dict(type='RandomFlip'),
                    dict(type='Normalize', **img_norm_cfg),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'],
                         meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape',
                            'scale_factor', 'flip', 'img_norm_cfg', 'is_first')),
                ])

        """
        img_info = self.img_infos[idx]
        frame_ind = img_info['frame_ind']
        foldername = img_info['foldername']
        ann_info = self.get_ann_info(idx, frame_ind)
        filename = osp.join(foldername, f"{frame_ind:06d}.JPEG")
        img_info['filename'] = filename
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        results['frame_ind'] = frame_ind
        results = self.pipeline(results)
        return results
