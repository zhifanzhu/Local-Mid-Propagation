import argparse
import os
import os.path as osp
import glob
import re
import numpy as np
import pickle

import mmcv
import torch
from mmcv.parallel import MMDataParallel
from mmcv.runner import get_dist_info, load_checkpoint

from mmdet.apis import init_dist
from mmdet.core import eval_map, results2json, wrap_fp16_model
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector


"""
test_video for specific video, or first n videos.

To test for ALL videos, please use test_video.py
"""


def single_gpu_test(model, data_loader, bgn, end, show=False):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(end - bgn)
    for i, data in enumerate(data_loader):
        if i < bgn:
            continue
        if i >= end:
            break
        with torch.no_grad():
            result = model(return_loss=False, rescale=not show, **data)
        results.append(result)

        if show:
            model.module.show_result(data, result, dataset.img_norm_cfg)

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def get_pascal_gts(dataset, bgn, end):
    gt_bboxes = []
    gt_labels = []
    gt_ignore = []
    for i in range(bgn, end):
        img_info = dataset.img_infos[i]
        frame_ind = img_info['frame_ind']
        ann = dataset.get_ann_info(i, frame_ind)
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
        gt_ignore = gt_ignore
    if hasattr(dataset, 'year') and dataset.year == 2007:
        dataset_name = 'voc07'
    elif hasattr(dataset, 'DATASET_NAME'):
        dataset_name = dataset.DATASET_NAME
    else:
        dataset_name = dataset.CLASSES
    return gt_bboxes, gt_labels, gt_ignore, dataset_name


def parse_args():

    def _str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--start', type=int, default=0, help='index from 0')
    parser.add_argument(
        '--num-videos', type=int, default=1, help='number of video to eval')
    parser.add_argument(
        '--shuffle', type=_str2bool, default=False,
        help='whether shuffle eval dataset')
    parser.add_argument(
        '--json_out',
        help='output result file name without extension',
        type=str)
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        choices=['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'],
        help='eval types')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def get_split_inds():
    #  Get split point, e.g. [0,0,0,1,1,2] -> [0,3,5,6]
    val_txt = 'data/ILSVRC2015/ImageSets/VID/VID_val_frames.txt'
    with open(val_txt) as fp:
        vids = fp.readlines()
    vids = ['/'.join(v.split(' ')[0].split('/')[:-1]) for v in vids]

    last_v = vids[0]
    split_ind = [0]
    for i in range(1, len(vids)):
        v = vids[i]
        if v != last_v:
            split_ind.append(i)
            last_v = v
    split_ind.append(len(vids))
    assert len(split_ind) == 556   # num val = 555
    return split_ind


def main():
    args = parse_args()

    # assert args.show or args.json_out, \
    #     ('Please specify at least one operation (save or show the results) '
    #      'with the argument "--out" or "--show" or "--json_out"')

    if args.json_out is not None and args.json_out.endswith('.json'):
        args.json_out = args.json_out[:-5]

    cfg = mmcv.Config.fromfile(args.config)

    """ Retrieve Checkpoint file """
    checkpoint_file = args.checkpoint
    if not checkpoint_file:
        def _epoch_num(name):
            return int(re.findall('epoch_[0-9]*.pth', name)[0].replace(
                'epoch_', '').replace('.pth', ''))
        pths = sorted(glob.glob(
            os.path.join(cfg.work_dir, 'epoch_*.pth')
        ), key=_epoch_num)
        if len(pths) > 0:
            print("Found {}, use it as checkpoint by default.".format(pths[-1]))
            checkpoint_file = pths[-1]
    if not checkpoint_file:
        raise ValueError("Checkpoints not found, check work_dir non empty.")
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True


    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # Build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=args.shuffle)  # TODO: hack shuffle True

    # Build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, checkpoint_file, map_location='cpu')

    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES


    split_inds = get_split_inds()
    bgn = split_inds[args.start]
    end = split_inds[args.start + args.num_videos]
    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, bgn, end, args.show)

    rank, _ = get_dist_info()
    if rank == 0:
        gt_bboxes, gt_labels, gt_ignore, dataset_name = get_pascal_gts(
            dataset, bgn, end)
        print('\nStarting evaluate {}'.format(dataset_name))
        eval_map(outputs, gt_bboxes, gt_labels, gt_ignore,
                 scale_ranges=None, iou_thr=0.5, dataset=dataset_name,
                 print_summary=True)


    # Always output to pkl for analysing.
    if args.out:
        with open(args.out, 'wb') as f:
            pickle.dump(outputs, f, pickle.HIGHEST_PROTOCOL)
    #     args.out = osp.join(
    #         cfg.work_dir,
    #         args.config.split('/')[-1].replace('.py', '_results.pkl'))
    # Save predictions in the COCO json format
    if args.json_out and rank == 0:
        if not isinstance(outputs[0], dict):
            results2json(dataset, outputs, args.json_out)
        else:
            for name in outputs[0]:
                outputs_ = [out[name] for out in outputs]
                result_file = args.json_out + '.{}'.format(name)
                results2json(dataset, outputs_, result_file)


if __name__ == '__main__':
    main()

