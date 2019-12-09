import torch
from mmdet.core.anchor import anchor_inside_flags
from mmdet.core.bbox import PseudoSampler, assign_and_sample, bbox2delta, build_assigner
from mmdet.core.anchor.anchor_target import images_to_levels, unmap
from mmdet.core.utils import multi_apply

def landm2delta(proposals, gt, means=[0, 0, 0, 0], stds=[1, 1, 1, 1]):
    assert proposals.size(0) == gt.size(0)
    batch_size = gt.size(0)
    gt = torch.reshape(gt, (batch_size, 5, 2))
    proposals = proposals.float()
    gt = gt.float()
    px = (proposals[..., 0] + proposals[..., 2]) * 0.5
    py = (proposals[..., 1] + proposals[..., 3]) * 0.5
    pw = proposals[..., 2] - proposals[..., 0] + 1.0
    ph = proposals[..., 3] - proposals[..., 1] + 1.0

    px = px.unsqueeze(1).expand(batch_size, 5)
    py = py.unsqueeze(1).expand(batch_size, 5)
    pw = pw.unsqueeze(1).expand(batch_size, 5)
    ph = ph.unsqueeze(1).expand(batch_size, 5)

    gx = gt[..., 0]
    gy = gt[..., 1]
    dx = (gx - px) / pw
    dy = (gy - py) / ph
    deltas = torch.stack([dx, dy], dim=-1)

    means = deltas.new_tensor(means[:2]).unsqueeze(0).expand(5, 2).unsqueeze(0)
    stds = deltas.new_tensor(stds[:2]).unsqueeze(0).expand(5, 2).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)

    deltas = deltas.reshape(batch_size, -1)
    return deltas

def retinaface_anchor_target(anchor_list,
                  valid_flag_list,
                  gt_bboxes_list,
                  gt_landms_list,
                  img_metas,
                  target_means,
                  target_stds,
                  cfg,
                  gt_bboxes_ignore_list=None,
                  gt_labels_list=None,
                  label_channels=1,
                  sampling=True,
                  unmap_outputs=True):
    """Compute regression and classification targets for anchors.

    Args:
        anchor_list (list[list]): Multi level anchors of each image.
        valid_flag_list (list[list]): Multi level valid flags of each image.
        gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
        gt_landms_list (list[Tensor]): Ground truth landmarks of each image.
        img_metas (list[dict]): Meta info of each image.
        target_means (Iterable): Mean value of regression targets.
        target_stds (Iterable): Std value of regression targets.
        cfg (dict): RPN train configs.

    Returns:
        tuple
    """
    num_imgs = len(img_metas)
    assert len(anchor_list) == len(valid_flag_list) == num_imgs

    # anchor number of multi levels
    num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
    # concat all level anchors and flags to a single tensor
    for i in range(num_imgs):
        assert len(anchor_list[i]) == len(valid_flag_list[i])
        anchor_list[i] = torch.cat(anchor_list[i])
        valid_flag_list[i] = torch.cat(valid_flag_list[i])

    # compute targets for each image
    if gt_bboxes_ignore_list is None:
        gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
    if gt_labels_list is None:
        gt_labels_list = [None for _ in range(num_imgs)]
    (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
     all_landm_targets, all_landm_weights, pos_inds_list,
     neg_inds_list) = multi_apply(
         retinaface_anchor_target_single,
         anchor_list,
         valid_flag_list,
         gt_bboxes_list,
         gt_bboxes_ignore_list,
         gt_landms_list,
         gt_labels_list,
         img_metas,
         target_means=target_means,
         target_stds=target_stds,
         cfg=cfg,
         label_channels=label_channels,
         sampling=sampling,
         unmap_outputs=unmap_outputs)
    # no valid anchors
    if any([labels is None for labels in all_labels]):
        return None
    # sampled anchors of all images
    num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
    num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
    # split targets to a list w.r.t. multiple levels
    labels_list = images_to_levels(all_labels, num_level_anchors)
    label_weights_list = images_to_levels(all_label_weights, num_level_anchors)
    bbox_targets_list = images_to_levels(all_bbox_targets, num_level_anchors)
    bbox_weights_list = images_to_levels(all_bbox_weights, num_level_anchors)
    landm_targets_list = images_to_levels(all_landm_targets, num_level_anchors)
    landm_weights_list = images_to_levels(all_landm_weights, num_level_anchors)
    return (labels_list, label_weights_list, bbox_targets_list,
            bbox_weights_list, landm_targets_list, landm_weights_list,
            num_total_pos, num_total_neg)


def retinaface_anchor_target_single(flat_anchors,
                         valid_flags,
                         gt_bboxes,
                         gt_bboxes_ignore,
                         gt_landms,
                         gt_labels,
                         img_meta,
                         target_means,
                         target_stds,
                         cfg,
                         label_channels=1,
                         sampling=True,
                         unmap_outputs=True):
    inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                       img_meta['img_shape'][:2],
                                       cfg.allowed_border)
    if not inside_flags.any():
        return (None, ) * 6
    # assign gt and sample anchors
    anchors = flat_anchors[inside_flags, :]

    if sampling:
        assign_result, sampling_result = assign_and_sample(
            anchors, gt_bboxes, gt_bboxes_ignore, None, cfg)
    else:
        bbox_assigner = build_assigner(cfg.assigner)
        assign_result = bbox_assigner.assign(anchors, gt_bboxes,
                                             gt_bboxes_ignore, gt_labels)
        bbox_sampler = PseudoSampler()
        sampling_result = bbox_sampler.sample(assign_result, anchors,
                                              gt_bboxes)

    num_valid_anchors = anchors.shape[0]
    bbox_targets = torch.zeros_like(anchors)
    bbox_weights = torch.zeros_like(anchors)
    labels = anchors.new_zeros(num_valid_anchors, dtype=torch.long)
    label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)
    landms_targets = anchors.new_zeros([num_valid_anchors, 10], dtype=torch.float)
    landms_weights = anchors.new_zeros([num_valid_anchors, 10], dtype=torch.float)

    pos_inds = sampling_result.pos_inds
    neg_inds = sampling_result.neg_inds
    if len(pos_inds) > 0:
        pos_bbox_targets = bbox2delta(sampling_result.pos_bboxes,
                                      sampling_result.pos_gt_bboxes,
                                      target_means, target_stds)
        bbox_targets[pos_inds, :] = pos_bbox_targets
        bbox_weights[pos_inds, :] = 1.0
        pos_gt_landms = gt_landms[sampling_result.pos_assigned_gt_inds, :]
        pos_landm_targets = landm2delta(sampling_result.pos_bboxes,
                                        pos_gt_landms, target_means,
                                        target_stds)
        landms_targets[pos_inds, :] = pos_landm_targets
        landms_weights[pos_inds, :] = 1.0
        if gt_labels is None:
            labels[pos_inds] = 1
        else:
            labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        if cfg.pos_weight <= 0:
            label_weights[pos_inds] = 1.0
        else:
            label_weights[pos_inds] = cfg.pos_weight
    if len(neg_inds) > 0:
        label_weights[neg_inds] = 1.0

    # map up to original set of anchors
    if unmap_outputs:
        num_total_anchors = flat_anchors.size(0)
        labels = unmap(labels, num_total_anchors, inside_flags)
        label_weights = unmap(label_weights, num_total_anchors, inside_flags)
        bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
        bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)
        landms_targets = unmap(landms_targets, num_total_anchors, inside_flags)
        landms_weights = unmap(landms_weights, num_total_anchors, inside_flags)

    return (labels, label_weights, bbox_targets, bbox_weights, landms_targets,
            landms_weights, pos_inds, neg_inds)
