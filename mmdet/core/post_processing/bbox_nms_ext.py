import torch

from mmdet.ops.nms import nms_wrapper


""" (ktw361)
Extension function(s) to original bbox_nms.py
"""


def multiclass_nms_with_feat(multi_bboxes,
                             multi_scores,
                             multi_feats,
                             score_thr,
                             nms_cfg,
                             max_num=-1,
                             score_factors=None):
    """NMS for multi-class bboxes.
    Returns:
        tuple: (bboxes, labels, embed_feat),
            tensors of shape (k, 5) and (k, 1) and (k, C),
            Labels are 0-based.
    """
    num_classes = multi_scores.shape[1]
    bboxes, labels, ret_feats = [], [], []
    ret_feat_channs = multi_feats.size(1)
    nms_cfg_ = nms_cfg.copy()
    nms_type = nms_cfg_.pop('type', 'nms')
    nms_op = getattr(nms_wrapper, nms_type)
    for i in range(1, num_classes):
        cls_inds = multi_scores[:, i] > score_thr
        if not cls_inds.any():
            continue
        # get bboxes and scores of this class
        if multi_bboxes.shape[1] == 4:
            _bboxes = multi_bboxes[cls_inds, :]
        else:
            _bboxes = multi_bboxes[cls_inds, i * 4:(i + 1) * 4]
        feats = multi_feats[cls_inds, :]
        _scores = multi_scores[cls_inds, i]
        if score_factors is not None:
            _scores *= score_factors[cls_inds]
        cls_dets = torch.cat([_bboxes, _scores[:, None]], dim=1)
        cls_dets, nms_inds = nms_op(cls_dets, **nms_cfg_)
        cls_labels = multi_bboxes.new_full((cls_dets.shape[0], ),
                                           i - 1,
                                           dtype=torch.long)
        bboxes.append(cls_dets)
        labels.append(cls_labels)
        ret_feats.append(feats[nms_inds])
    if bboxes:
        bboxes = torch.cat(bboxes)
        labels = torch.cat(labels)
        ret_feats = torch.cat(ret_feats)
        if bboxes.shape[0] > max_num:
            _, inds = bboxes[:, -1].sort(descending=True)
            inds = inds[:max_num]
            bboxes = bboxes[inds]
            labels = labels[inds]
            ret_feats = ret_feats[inds]
    else:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)
        ret_feats = multi_bboxes.new_zeros(
            (0, ret_feat_channs), dtype=torch.long)

    return bboxes, labels, ret_feats
