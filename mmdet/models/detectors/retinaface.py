from ..registry import DETECTORS
from .single_stage import SingleStageDetector


@DETECTORS.register_module
class RetinaFace(SingleStageDetector):
    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(RetinaFace, self).__init__(backbone, neck, bbox_head, train_cfg,
                                         test_cfg, pretrained)

    def forward_train(self,
                      img,
                      img_metas,
                      boxes,
                      landms,
                      labels,
                      gt_bboxes_ignore=None):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)

        loss_inputs = outs + (boxes, landms, labels, img_metas, self.train_cfg)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses
