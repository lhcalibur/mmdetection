import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init

from ..registry import HEADS
from ..builder import build_loss
from ..utils import ConvModule, bias_init_with_prob
from .anchor_head import AnchorHead
from mmdet.core import force_fp32, multi_apply, delta2bbox, delta2landm, retinaface_anchor_target
from mmdet.ops.nms import nms_wrapper


@HEADS.register_module
class RetinaFaceHead(AnchorHead):
    def __init__(self,
                 in_channels,
                 feat_channels=256,
                 loss_landm=dict(
                     type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
                 **kwargs):
        num_classes = 2
        super(RetinaFaceHead, self).__init__(num_classes, in_channels, feat_channels, **kwargs)
        self.loss_landm = build_loss(loss_landm)

    def _init_layers(self):
        self.ssh = SSH(self.in_channels, self.feat_channels)
        self.face_cls = nn.Conv2d(self.feat_channels,
                                  self.num_anchors * self.cls_out_channels, 1)
        self.bbox_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 4, 1)
        self.ladm_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 10, 1)

    def init_weights(self):
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.face_cls, std=0.01, bias=bias_cls)
        normal_init(self.bbox_reg, std=0.01)
        normal_init(self.ladm_reg, std=0.01)

    def forward_single(self, x):
        x = self.ssh(x)

        face_cls = self.face_cls(x)
        bbox_reg = self.bbox_reg(x)
        ladm_reg = self.ladm_reg(x)
        return face_cls, bbox_reg, ladm_reg

    def loss_single(self, cls_score, bbox_pred, landm_pred, labels, label_weights,
                    bbox_targets, bbox_weights, landms_targets, landms_weights,
                    num_total_samples, cfg):
        loss_cls, loss_bbox = super(RetinaFaceHead, self).loss_single(
            cls_score,
            bbox_pred,
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            num_total_samples,
            cfg)
        # landmarks loss
        landms_targets = landms_targets.reshape(-1, 10)
        landms_weights = landms_weights.reshape(-1, 10)
        landm_pred = landm_pred.permute(0, 2, 3, 1).reshape(-1, 10)
        loss_landm = self.loss_landm(
            landm_pred,
            landms_targets,
            landms_weights,
            avg_factor=num_total_samples)
        return loss_cls, loss_bbox, loss_landm

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'landm_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             landm_preds,
             gt_bboxes,
             gt_landms,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == len(self.anchor_generators)

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = retinaface_anchor_target(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            gt_landms,
            img_metas,
            self.target_means,
            self.target_stds,
            cfg,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            sampling=self.sampling)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         landm_targets_list, landm_weights_list, num_total_pos,
         num_total_neg) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)
        losses_cls, losses_bbox, losses_landm = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            landm_preds,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            landm_targets_list,
            landm_weights_list,
            num_total_samples=num_total_samples,
            cfg=cfg)
        return dict(
            loss_cls=losses_cls,
            loss_bbox=losses_bbox,
            losses_landms=losses_landm)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'landm_preds'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   landm_preds,
                   img_metas,
                   cfg,
                   rescale=False):
        """
        Transform network output for a batch into labeled boxes.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            landm_preds (list[Tensor]): Landmarks
            img_metas (list[dict]): size / scale info for each image
            cfg (mmcv.Config): test / postprocessing configuration
            rescale (bool): if True, return boxes in original image space

        Returns:
            list[tuple[Tensor, Tensor]]: each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the class index of the
                corresponding box.
        """
        assert len(cls_scores) == len(bbox_preds) == len(landm_preds)
        num_levels = len(cls_scores)

        device = cls_scores[0].device
        mlvl_anchors = [
            self.anchor_generators[i].grid_anchors(
                cls_scores[i].size()[-2:],
                self.anchor_strides[i],
                device=device) for i in range(num_levels)
        ]
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            landm_pred_list = [
                landm_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self.get_bboxes_single(cls_score_list, bbox_pred_list,
                                               landm_pred_list, mlvl_anchors,
                                               img_shape, scale_factor, cfg,
                                               rescale)
            result_list.append(proposals)
        return result_list

    def get_bboxes_single(self,
                          cls_score_list,
                          bbox_pred_list,
                          landm_pred_list,
                          mlvl_anchors,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        """
        Transform outputs for a single batch item into labeled boxes.
        """
        assert len(cls_score_list) == len(bbox_pred_list) == len(
            landm_pred_list) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_landms = []
        mlvl_scores = []
        for cls_score, bbox_pred, landm_pred, anchors in zip(
                cls_score_list, bbox_pred_list, landm_pred_list, mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:] == landm_pred.size()[-2:]
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            landm_pred = landm_pred.permute(1, 2, 0).reshape(-1, 10)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                # Get maximum scores for foreground classes.
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    max_scores, _ = scores[:, 1:].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                landm_pred = landm_pred[topk_inds, :]
                scores = scores[topk_inds, :]
            bboxes = delta2bbox(anchors, bbox_pred, self.target_means,
                                self.target_stds, img_shape)
            landms = delta2landm(anchors, landm_pred, self.target_means,
                                 self.target_stds, img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_landms.append(landms)
            mlvl_scores.append(scores)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_landms = torch.cat(mlvl_landms)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
            mlvl_landms /= mlvl_landms.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        if self.use_sigmoid_cls:
            # Add a dummy background class to the front when using sigmoid
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)
        det_bboxes, det_landms, det_labels = multiclass_nms(mlvl_bboxes, mlvl_landms,
                                                mlvl_scores, cfg.score_thr,
                                                cfg.nms, cfg.max_per_img)
        return det_bboxes, det_landms, det_labels


def multiclass_nms(multi_bboxes,
                   multi_landms,
                   multi_scores,
                   score_thr,
                   nms_cfg,
                   max_num=-1,
                   score_factors=None):
    assert multi_bboxes.shape[1] == 4
    num_classes = multi_scores.shape[1]
    bboxes, landms, labels = [], [], []
    nms_cfg_ = nms_cfg.copy()
    nms_type = nms_cfg_.pop('type', 'nms')
    nms_op = getattr(nms_wrapper, nms_type)
    for i in range(1, num_classes):
        cls_inds = multi_scores[:, i] > score_thr
        if not cls_inds.any():
            continue
        # get bboxes and scores of this class
        _bboxes = multi_bboxes[cls_inds, :]
        _landms = multi_landms[cls_inds, :]
        _scores = multi_scores[cls_inds, i]
        if score_factors is not None:
            _scores *= score_factors[cls_inds]
        cls_dets = torch.cat([_bboxes, _scores[:, None]], dim=1)
        cls_dets, inds = nms_op(cls_dets, **nms_cfg_)
        cls_labels = multi_bboxes.new_full((cls_dets.shape[0], ),
                                           i - 1,
                                           dtype=torch.long)
        bboxes.append(cls_dets)
        landms.append(_landms[inds])
        labels.append(cls_labels)
    if bboxes:
        bboxes = torch.cat(bboxes)
        landms = torch.cat(landms)
        labels = torch.cat(labels)
        if bboxes.shape[0] > max_num:
            _, inds = bboxes[:, -1].sort(descending=True)
            inds = inds[:max_num]
            bboxes = bboxes[inds]
            landms = landms[inds]
            labels = labels[inds]
    else:
        bboxes = multi_bboxes.new_zeros((0, 5))
        landms = multi_landms.new_zeros((0, 10))
        labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)

    return bboxes, landms, labels


def conv_bn(inp, oup, stride=1, leaky=0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )


def conv_bn_no_relu(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
    )


class SSH(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SSH, self).__init__()
        assert out_channel % 4 == 0
        leaky = 0
        if out_channel <= 64:
            leaky = 0.1
        self.conv3X3 = conv_bn_no_relu(in_channel, out_channel // 2, stride=1)

        self.conv5X5_1 = conv_bn(in_channel, out_channel // 4, stride=1, leaky=leaky)
        self.conv5X5_2 = conv_bn_no_relu(out_channel // 4, out_channel // 4, stride=1)

        self.conv7X7_2 = conv_bn(out_channel // 4, out_channel // 4, stride=1, leaky=leaky)
        self.conv7x7_3 = conv_bn_no_relu(out_channel // 4, out_channel // 4, stride=1)

    def forward(self, input):
        conv3X3 = self.conv3X3(input)

        conv5X5_1 = self.conv5X5_1(input)
        conv5X5 = self.conv5X5_2(conv5X5_1)

        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7x7_3(conv7X7_2)

        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)
        out = F.relu(out)
        return out