import os
import os.path as osp

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.parallel import collate, scatter
from mmcv.runner import Hook
from pycocotools.cocoeval import COCOeval
from torch.utils.data import Dataset

from mmdet import datasets
from .coco_utils import fast_eval_recall, results2json
from .mean_ap import eval_map


# TODO: fix num_evals and shuffle for COCO

class DistEvalHook(Hook):

    def __init__(self, dataset, interval=1, num_evals=-1, shuffle=False):
        if isinstance(dataset, Dataset):
            self.dataset = dataset
        elif isinstance(dataset, dict):
            self.dataset = datasets.build_dataset(dataset, {'test_mode': True})
        else:
            raise TypeError(
                'dataset must be a Dataset object or a dict, not {}'.format(
                    type(dataset)))
        self.interval = interval
        self.num_evals = num_evals
        if self.num_evals < 0:
            self.num_evals = len(self.dataset)
        self.shuffle = shuffle
        if hasattr(self.dataset, 'coco') and self.shuffle and self.num_evals > 0:
            raise NotImplementedError

    def after_train_epoch(self, runner):
        if not self.every_n_epochs(runner, self.interval):
            return
        runner.model.eval()
        range_idxs = list(range(runner.rank, len(self.dataset), runner.world_size))
        if self.shuffle:
            np.random.shuffle(range_idxs)
        range_idxs = range_idxs[:self.num_evals]
        if runner.rank == 0:
            prog_bar = mmcv.ProgressBar(len(range_idxs) * runner.world_size)
        results = []
        for idx in range_idxs:
            data = self.dataset[idx]
            data_gpu = scatter(
                collate([data], samples_per_gpu=1),
                [torch.cuda.current_device()])[0]

            # compute output
            with torch.no_grad():
                result = runner.model(
                    return_loss=False, rescale=True, **data_gpu)
            results.append(result)

            batch_size = runner.world_size
            if runner.rank == 0:
                for _ in range(batch_size):
                    prog_bar.update()

        if runner.rank == 0:
            print('\n')
            dist.barrier()
            for i in range(1, runner.world_size):
                tmp_file = osp.join(runner.work_dir, 'temp_{}.pkl'.format(i))
                tmp_range = osp.join(runner.work_dir, 'temp_range_{}.pkl'.format(i))
                tmp_results = mmcv.load(tmp_file)
                tmp_range_idxs = mmcv.load(tmp_range)
                results.extend(tmp_results)
                range_idxs.extend(tmp_range_idxs)
                os.remove(tmp_file)
                os.remove(tmp_range)
            self.evaluate(runner, results, range_idxs)
        else:
            tmp_file = osp.join(runner.work_dir,
                                'temp_{}.pkl'.format(runner.rank))
            tmp_range = osp.join(runner.work_dir,
                                 'temp_range_{}.pkl'.format(runner.rank))
            mmcv.dump(results, tmp_file)
            mmcv.dump(range_idxs, tmp_range)
            dist.barrier()
        dist.barrier()

    def evaluate(self):
        raise NotImplementedError


class DistEvalmAPHook(DistEvalHook):

    def evaluate(self, runner, results, range_idxs=None):
        gt_bboxes = []
        gt_labels = []
        gt_ignore = []
        if range_idxs is None:
            range_idxs = range(len(self.dataset))
        for i in range_idxs:
            ann = self.dataset.get_ann_info(i)
            bboxes = ann['bboxes']
            labels = ann['labels']
            if 'bboxes_ignore' in ann:
                ignore = np.concatenate([
                    np.zeros(bboxes.shape[0], dtype=np.bool),
                    np.ones(ann['bboxes_ignore'].shape[0], dtype=np.bool)
                ])
                gt_ignore.append(ignore)
                bboxes = np.vstack([bboxes, ann['bboxes_ignore']])
                labels = np.concatenate([labels, ann['labels_ignore']])
            gt_bboxes.append(bboxes)
            gt_labels.append(labels)
        if not gt_ignore:
            gt_ignore = None
        # If the dataset is VOC2007, then use 11 points mAP evaluation.
        if hasattr(self.dataset, 'year') and self.dataset.year == 2007:
            ds_name = 'voc07'
        elif hasattr(self.dataset, 'DATASET_NAME'):
            ds_name = self.dataset.DATASET_NAME
        else:
            ds_name = self.dataset.CLASSES
        mean_ap, eval_results = eval_map(
            results,
            gt_bboxes,
            gt_labels,
            gt_ignore=gt_ignore,
            scale_ranges=None,
            iou_thr=0.5,
            dataset=ds_name,
            print_summary=True)
        runner.log_buffer.output['mAP'] = mean_ap
        runner.log_buffer.ready = True


class CocoDistEvalRecallHook(DistEvalHook):

    def __init__(self,
                 dataset,
                 interval=1,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=np.arange(0.5, 0.96, 0.05)):
        super(CocoDistEvalRecallHook, self).__init__(
            dataset, interval=interval)
        self.proposal_nums = np.array(proposal_nums, dtype=np.int32)
        self.iou_thrs = np.array(iou_thrs, dtype=np.float32)

    def evaluate(self, runner, results):
        # the official coco evaluation is too slow, here we use our own
        # implementation instead, which may get slightly different results
        ar = fast_eval_recall(results, self.dataset.coco, self.proposal_nums,
                              self.iou_thrs)
        for i, num in enumerate(self.proposal_nums):
            runner.log_buffer.output['AR@{}'.format(num)] = ar[i]
        runner.log_buffer.ready = True


class CocoDistEvalmAPHook(DistEvalHook):

    def evaluate(self, runner, results):
        tmp_file = osp.join(runner.work_dir, 'temp_0')
        result_files = results2json(self.dataset, results, tmp_file)

        res_types = ['bbox', 'segm'
                     ] if runner.model.module.with_mask else ['bbox']
        cocoGt = self.dataset.coco
        imgIds = cocoGt.getImgIds()
        for res_type in res_types:
            try:
                cocoDt = cocoGt.loadRes(result_files[res_type])
            except IndexError:
                print('No prediction found.')
                break
            iou_type = res_type
            cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
            cocoEval.params.imgIds = imgIds
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            # metrics = ['mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l']
            metrics = [
                'Precision/mAP', 'Precision/mAP_.50IOU', 'Precision/mAP_.75IOU',
                'Precision/mAP_small', 'Precision/mAP_medium', 'Precision/mAP_large',
                'Recall/AR_1', 'Recall/AR_10', 'Recall/AR_100', 'Recall/AR_100_small',
                'Recall/AR_100_medium', 'Recall/AR_100_large']
            for i in range(len(metrics)):
                key = '{}_{}'.format(res_type, metrics[i])
                val = float('{:.3f}'.format(cocoEval.stats[i]))
                runner.log_buffer.output[key] = val
            runner.log_buffer.output['{}_mAP_copypaste'.format(res_type)] = (
                '{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                '{ap[4]:.3f} {ap[5]:.3f}').format(ap=cocoEval.stats[:6])
        runner.log_buffer.ready = True
        for res_type in res_types:
            os.remove(result_files[res_type])


class NonDistEvalHook(Hook):

    def __init__(self, dataset, interval=1, num_evals=-1, shuffle=False):
        if isinstance(dataset, Dataset):
            self.dataset = dataset
        elif isinstance(dataset, dict):
            self.dataset = datasets.build_dataset(dataset, {'test_mode': True})
        else:
            raise TypeError(
                'dataset must be a Dataset object or a dict, not {}'.format(
                    type(dataset)))
        self.interval = interval
        self.num_evals = num_evals
        if self.num_evals < 0:
            self.num_evals = len(self.dataset)
        self.shuffle = shuffle
        if hasattr(self.dataset, 'coco') and self.shuffle and self.num_evals > 0:
            raise NotImplementedError

    def after_train_epoch(self, runner):
        if not self.every_n_epochs(runner, self.interval):
            return
        runner.model.eval()
        range_idxs = list(range(len(self.dataset)))
        if self.shuffle:
            np.random.shuffle(range_idxs)
        range_idxs = range_idxs[:self.num_evals]
        prog_bar = mmcv.ProgressBar(len(range_idxs))
        results = []
        for idx in range_idxs:
            data = self.dataset[idx]
            data_gpu = scatter(
                collate([data], samples_per_gpu=1),
                [torch.cuda.current_device()])[0]

            with torch.no_grad():
                result = runner.model(
                    return_loss=False, rescale=True, **data_gpu)
            results.append(result)

            prog_bar.update()

        self.evaluate(runner, results, range_idxs=range_idxs)

    def evaluate(self):
        raise NotImplementedError


class NonDistEvalmAPHook(NonDistEvalHook):

    def evaluate(self, runner, results, range_idxs=None):
        gt_bboxes = []
        gt_labels = []
        gt_ignore = []
        if range_idxs is None:
            range_idxs = range(len(self.dataset))
        for i in range_idxs:
            ann = self.dataset.get_ann_info(i)
            bboxes = ann['bboxes']
            labels = ann['labels']
            if gt_ignore is not None:
                ignore = np.concatenate([
                    np.zeros(bboxes.shape[0], dtype=np.bool),
                    np.ones(ann['bboxes_ignore'].shape[0], dtype=np.bool)
                ])
                gt_ignore.append(ignore)
                bboxes = np.vstack([bboxes, ann['bboxes_ignore']])
                labels = np.concatenate([labels, ann['labels_ignore']])
            gt_bboxes.append(bboxes)
            gt_labels.append(labels)
        # If the dataset is VOC2007, then use 11 points mAP evaluation.
        if hasattr(self.dataset, 'year') and self.dataset.year == 2007:
            ds_name = 'voc07'
        elif hasattr(self.dataset, 'DATASET_NAME'):
            ds_name = self.dataset.DATASET_NAME
        else:
            ds_name = self.dataset.CLASSES
        mean_ap, eval_results = eval_map(
            results,
            gt_bboxes,
            gt_labels,
            gt_ignore=gt_ignore,
            scale_ranges=None,
            iou_thr=0.5,
            dataset=ds_name,
            print_summary=True)
        runner.log_buffer.output['mAP'] = mean_ap
        runner.log_buffer.ready = True


class CocoNonDistEvalmAPHook(NonDistEvalHook):

    def evaluate(self, runner, results):
        tmp_file = osp.join(runner.work_dir, 'temp_0')
        result_files = results2json(self.dataset, results, tmp_file)

        res_types = ['bbox',
                     'segm'] if runner.model.module.with_mask else ['bbox']
        cocoGt = self.dataset.coco
        imgIds = cocoGt.getImgIds()
        for res_type in res_types:
            cocoDt = cocoGt.loadRes(result_files[res_type])
            iou_type = res_type
            cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
            cocoEval.params.imgIds = imgIds
            # cocoEval.params.maxDets = [1, 10, 100, 500]  # (TODO)Set for visdrone
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            # metrics = ['mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l']
            metrics = [
                'Precision/mAP', 'Precision/mAP_.50IOU', 'Precision/mAP_.75IOU',
                'Precision/mAP_small', 'Precision/mAP_medium', 'Precision/mAP_large',
                'Recall/AR_1', 'Recall/AR_10', 'Recall/AR_100', 'Recall/AR_100_small',
                'Recall/AR_100_medium', 'Recall/AR_100_large']
            for i in range(len(metrics)):
                key = '{}_{}'.format(res_type, metrics[i])
                val = float('{:.3f}'.format(cocoEval.stats[i]))
                runner.log_buffer.output[key] = val
            runner.log_buffer.output['{}_mAP_copypaste'.format(res_type)] = (
                '{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                '{ap[4]:.3f} {ap[5]:.3f}').format(ap=cocoEval.stats[:6])
        runner.log_buffer.ready = True
        for res_type in res_types:
            os.remove(result_files[res_type])


class NonDistSeqEvalmAPHook(Hook):
    """ Currently, the only diff is results extend instead of append."""

    def __init__(self, dataset, interval=1, num_evals=-1, shuffle=False):
        if isinstance(dataset, Dataset):
            self.dataset = dataset
        elif isinstance(dataset, dict):
            self.dataset = datasets.build_dataset(dataset, {'test_mode': True})
        else:
            raise TypeError(
                'dataset must be a Dataset object or a dict, not {}'.format(
                    type(dataset)))
        self.interval = interval
        self.num_evals = num_evals
        if self.num_evals < 0:
            self.num_evals = len(self.dataset)
        self.shuffle = shuffle
        if hasattr(self.dataset, 'coco') and self.shuffle and self.num_evals > 0:
            raise NotImplementedError

    def after_train_epoch(self, runner):
        if not self.every_n_epochs(runner, self.interval):
            return
        runner.model.eval()
        range_idxs = list(range(len(self.dataset)))
        if self.shuffle:
            np.random.shuffle(range_idxs)
        range_idxs = range_idxs[:self.num_evals]
        prog_bar = mmcv.ProgressBar(len(range_idxs))
        results = []
        for idx in range_idxs:
            data = self.dataset[idx]
            data_gpu = scatter(
                collate([data], samples_per_gpu=1),
                [torch.cuda.current_device()])[0]

            with torch.no_grad():
                result, out_dict = runner.model(
                    return_loss=False, rescale=True, **data_gpu)
            results.extend(result)

            prog_bar.update()

        self.evaluate(runner, results, range_idxs=range_idxs)

    def evaluate(self, runner, results, range_idxs=None):
        """ If more than 'seq_len' frames per video, only first 'seq_len' will
        be loaded.
        """
        gt_bboxes = []
        gt_labels = []
        gt_ignore = []
        if range_idxs is None:
            range_idxs = range(len(self.dataset))
        for i in range_idxs:
            frame_ids = self.dataset.select_test_clip(i)
            anns = self.dataset.get_ann_info(i, frame_ids)
            for ann in anns:
                bboxes = ann['bboxes']
                labels = ann['labels']
                if gt_ignore is not None:
                    ignore = np.concatenate([
                        np.zeros(bboxes.shape[0], dtype=np.bool),
                        np.ones(ann['bboxes_ignore'].shape[0], dtype=np.bool)
                    ])
                    gt_ignore.append(ignore)
                    bboxes = np.vstack([bboxes, ann['bboxes_ignore']])
                    labels = np.concatenate([labels, ann['labels_ignore']])
                gt_bboxes.append(bboxes)
                gt_labels.append(labels)
        if hasattr(self.dataset, 'DATASET_NAME'):
            ds_name = self.dataset.DATASET_NAME
        else:
            ds_name = self.dataset.CLASSES
        mean_ap, eval_results = eval_map(
            results,
            gt_bboxes,
            gt_labels,
            gt_ignore=gt_ignore,
            scale_ranges=None,
            iou_thr=0.5,
            dataset=ds_name,
            print_summary=True)
        runner.log_buffer.output['mAP'] = mean_ap
        runner.log_buffer.ready = True


class NonDistPairEvalmAPHook(NonDistEvalHook):

    def __init__(self, dataset, interval=1, num_evals=-1, shuffle=False):
        super(NonDistPairEvalmAPHook, self).__init__(
                dataset, interval, num_evals, shuffle)
        assert self.shuffle is False, "Shuffle must be false in Pair mode"

    def evaluate(self, runner, results, range_idxs=None):
        gt_bboxes = []
        gt_labels = []
        gt_ignore = []
        if range_idxs is None:
            range_idxs = range(len(self.dataset))
        for i in range_idxs:
            img_info = self.dataset.img_infos[i]
            frame_ind = img_info['frame_ind']
            ann = self.dataset.get_ann_info(i, frame_ind)
            bboxes = ann['bboxes']
            labels = ann['labels']
            if gt_ignore is not None:
                ignore = np.concatenate([
                    np.zeros(bboxes.shape[0], dtype=np.bool),
                    np.ones(ann['bboxes_ignore'].shape[0], dtype=np.bool)
                ])
                gt_ignore.append(ignore)
                bboxes = np.vstack([bboxes, ann['bboxes_ignore']])
                labels = np.concatenate([labels, ann['labels_ignore']])
            gt_bboxes.append(bboxes)
            gt_labels.append(labels)
        if hasattr(self.dataset, 'DATASET_NAME'):
            ds_name = self.dataset.DATASET_NAME
        else:
            ds_name = self.dataset.CLASSES
        mean_ap, eval_results = eval_map(
            results,
            gt_bboxes,
            gt_labels,
            gt_ignore=gt_ignore,
            scale_ranges=None,
            iou_thr=0.5,
            dataset=ds_name,
            print_summary=True)
        runner.log_buffer.output['mAP'] = mean_ap
        runner.log_buffer.ready = True
