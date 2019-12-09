import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init

from ..registry import HEADS
from ..builder import build_loss
from ..utils import ConvModule, bias_init_with_prob
from .anchor_head import AnchorHead
from mmdet.core import force_fp32, multi_apply
from mmdet.core import retinaface_anchor_target


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