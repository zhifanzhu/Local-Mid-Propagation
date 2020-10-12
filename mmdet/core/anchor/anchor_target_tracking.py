import torch

from ..bbox import PseudoSampler, assign_and_sample, build_assigner
from ..utils import multi_apply

"""
2019.09.21
This was created for D&T
"""


def anchor_target_tracking(anchor_list,
                           valid_flag_list,
                           gt_bboxes_list,
                           gt_deltas_list,
                           img_metas,
                           cfg,
                           sampling=True,
                           unmap_outputs=True):
    """Compute regression and classification targets for anchors.

    Args:
        anchor_list (list[list]): Multi level anchors of each image.
        valid_flag_list (list[list]): Multi level valid flags of each image.
        gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
        gt_deltas_list (list[Tensor]):
        img_metas (list[dict]): Meta info of each image.
        cfg (dict): RPN train configs.

    Returns:
        tuple
    """
    num_imgs = len(img_metas)
    assert len(anchor_list) == len(valid_flag_list) == num_imgs

    # anchor number of multi levels
    num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
    # num_level_anchors : list [5776, 2166, 600, 150, 36, 4]
    # concat all level anchors and flags to a single tensor
    for i in range(num_imgs):
        assert len(anchor_list[i]) == len(valid_flag_list[i])
        anchor_list[i] = torch.cat(anchor_list[i])
        valid_flag_list[i] = torch.cat(valid_flag_list[i])

    # compute targets for each image
    (all_bbox_targets, all_bbox_weights,
     pos_inds_list, neg_inds_list) = multi_apply(
        anchor_target_tracking_single,
        anchor_list,
        valid_flag_list,
        gt_bboxes_list,
        gt_deltas_list,
        img_metas,
        cfg=cfg,
        sampling=sampling,
        unmap_outputs=unmap_outputs)
    # sampled anchors of all images
    num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
    num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
    # split targets to a list w.r.t. multiple levels
    bbox_targets_list = images_to_levels(all_bbox_targets, num_level_anchors)
    bbox_weights_list = images_to_levels(all_bbox_weights, num_level_anchors)
    return (bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg)


def images_to_levels(target, num_level_anchors):
    """Convert targets by image to targets by feature level.

    [target_img0, target_img1] -> [target_level0, target_level1, ...]
    """
    target = torch.stack(target, 0)
    level_targets = []
    start = 0
    for n in num_level_anchors:
        end = start + n
        level_targets.append(target[:, start:end].squeeze(0))
        start = end
    return level_targets


def anchor_target_tracking_single(flat_anchors,
                                  valid_flags,
                                  gt_bboxes,
                                  gt_deltas,
                                  img_meta,
                                  cfg,
                                  sampling=True,
                                  unmap_outputs=True):
    """ Strategy: use gt_deltas as gt_bboxes, e.g. sample(), sample_result=gt_bboxes.
    """
    inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                       img_meta['img_shape'][:2],
                                       cfg.allowed_border)
    if not inside_flags.any():
        return (None, ) * 6
    # assign gt and sample anchors
    anchors = flat_anchors[inside_flags, :]

    if sampling:
        assign_result, sampling_result = assign_and_sample(
            anchors, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None, cfg=cfg)
    else:
        bbox_assigner = build_assigner(cfg.assigner)
        assign_result = bbox_assigner.assign(anchors, gt_bboxes,
                                             gt_bboxes_ignore=None,
                                             gt_labels=None)
        bbox_sampler = PseudoSampler()
        # sampling_result = bbox_sampler.sample(assign_result, anchors, gt_bboxes)
        sampling_result = bbox_sampler.sample(assign_result, anchors, gt_deltas)

    bbox_targets = torch.zeros_like(anchors)
    bbox_weights = torch.zeros_like(anchors)

    pos_inds = sampling_result.pos_inds
    neg_inds = sampling_result.neg_inds
    if len(pos_inds) > 0:
        # pos_bbox_targets = bbox2delta(sampling_result.pos_bboxes,
        #                               sampling_result.pos_gt_bboxes,
        #                               target_means, target_stds)
        pos_bbox_targets = sampling_result.pos_gt_bboxes
        bbox_targets[pos_inds, :] = pos_bbox_targets
        bbox_weights[pos_inds, :] = 1.0

    # map up to original set of anchors
    if unmap_outputs:
        num_total_anchors = flat_anchors.size(0)
        bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
        bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

    return bbox_targets, bbox_weights, pos_inds, neg_inds


def anchor_inside_flags(flat_anchors, valid_flags, img_shape,
                        allowed_border=0):
    img_h, img_w = img_shape[:2]
    if allowed_border >= 0:
        inside_flags = valid_flags & \
            (flat_anchors[:, 0] >= -allowed_border).type(torch.uint8) & \
            (flat_anchors[:, 1] >= -allowed_border).type(torch.uint8) & \
            (flat_anchors[:, 2] < img_w + allowed_border).type(torch.uint8) & \
            (flat_anchors[:, 3] < img_h + allowed_border).type(torch.uint8)
    else:
        inside_flags = valid_flags
    return inside_flags


def unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if data.dim() == 1:
        ret = data.new_full((count, ), fill)
        ret[inds] = data
    else:
        new_size = (count, ) + data.size()[1:]
        ret = data.new_full(new_size, fill)
        ret[inds, :] = data
    return ret
