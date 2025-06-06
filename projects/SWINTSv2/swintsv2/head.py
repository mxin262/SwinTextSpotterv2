import copy
import math
from typing import Optional, List

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from detectron2.modeling.poolers import ROIPooler, cat
from detectron2.structures import Boxes, pairwise_iou

from detectron2.layers import Conv2d, ConvTranspose2d, ShapeSpec, get_norm
from torchvision.ops.boxes import nms
from detectron2.modeling.matcher import Matcher
from .rec_stage import REC_STAGE   
from timm.models.layers import trunc_normal_, DropPath
_DEFAULT_SCALE_CLAMP = math.log(100000.0 / 16)

def _get_src_permutation_idx(indices):
# permute predictions following indices
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx

def _get_tgt_permutation_idx(indices):
# permute targets following indices
    batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
    tgt_idx = torch.cat([tgt for (_, tgt) in indices])
    return batch_idx, tgt_idx

class DynamicHead(nn.Module):

    def __init__(self, cfg, roi_input_shape):
        super().__init__()

        # Build RoI.
        box_pooler = self._init_box_pooler(cfg, roi_input_shape)
        self.box_pooler = box_pooler
        box_pooler_rec = self._init_box_pooler_rec(cfg, roi_input_shape)
        self.box_pooler_rec = box_pooler_rec

        # Build heads.
        num_classes = cfg.MODEL.SWINTS.NUM_CLASSES
        self.hidden_dim = cfg.MODEL.SWINTS.HIDDEN_DIM
        dim_feedforward = cfg.MODEL.SWINTS.DIM_FEEDFORWARD
        nhead = cfg.MODEL.SWINTS.NHEADS
        dropout = cfg.MODEL.SWINTS.DROPOUT
        activation = cfg.MODEL.SWINTS.ACTIVATION
        self.train_num_proposal = cfg.MODEL.SWINTS.NUM_PROPOSALS
        self.num_heads = cfg.MODEL.SWINTS.NUM_HEADS
        rcnn_head = RCNNHead(cfg, self.hidden_dim, num_classes, dim_feedforward, nhead, dropout, activation)
        self.head_series = _get_clones(rcnn_head, self.num_heads)
        self.return_intermediate = cfg.MODEL.SWINTS.DEEP_SUPERVISION
        
        self.cfg =cfg

        # Build recognition heads
        self.rec_stage = REC_STAGE(cfg, self.hidden_dim, num_classes, dim_feedforward, nhead, dropout, activation)
        
        # Init parameters.
        self.num_classes = num_classes
        prior_prob = cfg.MODEL.SWINTS.PRIOR_PROB
        self.bias_value = -math.log((1 - prior_prob) / prior_prob)
        self._reset_parameters()

    def _reset_parameters(self):
        # init all parameters.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

            # initialize the bias for focal loss.
            if p.shape[-1] == self.num_classes:
                nn.init.constant_(p, self.bias_value)

    @staticmethod
    def _init_box_pooler(cfg, input_shape):

        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        return box_pooler
    @staticmethod
    def _init_box_pooler_rec(cfg, input_shape):
        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = (cfg.MODEL.REC_HEAD.POOLER_RESOLUTION[0], cfg.MODEL.REC_HEAD.POOLER_RESOLUTION[1] *2)
        # pooler_resolution = (cfg.MODEL.REC_HEAD.POOLER_RESOLUTION[0], cfg.MODEL.REC_HEAD.POOLER_RESOLUTION[1])
        # pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
        # pooler_resolution = (28,56)
        pooler_scales = (0.25,)
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales= pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        return box_pooler
   
    def extra_rec_feat(self, matcher, mask_encoding, targets, N, bboxes, class_logits, pred_bboxes, mask_logits, proposal_features, features):
        gt_masks = list()
        gt_boxes = list()
        proposal_boxes_pred = list()
        masks_pred = list()
        pred_mask = mask_logits

        N, nr_boxes = bboxes.shape[:2]
        if targets:
            output = {'pred_logits': class_logits, 'pred_boxes': pred_bboxes, 'pred_masks': mask_logits}
            indices = matcher(output, targets, mask_encoding)
            idx = _get_src_permutation_idx(indices)
            target_rec = torch.cat([t['rec'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            # target_rec = target_rec.repeat(2,1)
        else:
            bs = 1
            # import pdb;pdb.set_trace()
            class_logits = torch.cat(class_logits)
            pred_bboxes = torch.cat(pred_bboxes)
            mask_logits = torch.cat(mask_logits)
            proposal_features = torch.cat(proposal_features)
            num_stage, n_queries, n_cls = class_logits.shape
            n_queries = n_queries * num_stage
            pred_bboxes = pred_bboxes.flatten(0,1)
            class_logits = class_logits.flatten(0,1)
            mask_logits = mask_logits.flatten(0,1)
            proposal_features = proposal_features.flatten(0,1)
            # scores = class_logits.sigmoid()
            # # all_scores = prob.view(bs, n_queries * n_cls).to(class_logits.device)
            # # all_indexes = torch.arange(n_queries * n_cls)[None].repeat(bs, 1).to(class_logits.device)
            # # all_labels = all_indexes % n_cls
            keep_inds = nms(pred_bboxes, class_logits[:,0].sigmoid(), 0.7)
            bboxes = pred_bboxes[keep_inds]
            class_logits = class_logits[keep_inds]
            mask_logits = mask_logits[keep_inds]
            proposal_features = proposal_features[keep_inds]
            idx = None
            # import pdb;pdb.set_trace()
            scores = torch.sigmoid(class_logits)
            labels = torch.arange(2, device=bboxes.device).\
                    unsqueeze(0).repeat(len(keep_inds), 1).flatten(0, 1)
            inter_class_logits = []
            inter_pred_bboxes = []
            inter_pred_masks = []
            inter_pred_label = []
        for b in range(N):
            if targets:
                gt_boxes.append(Boxes(targets[b]['boxes_xyxy'][indices[b][1]]))
                gt_masks.append(targets[b]['gt_masks'][indices[b][1]])
                # proposal_boxes_pred.append(Boxes(bboxes[b][indices[b][0]]))
                # tmp_mask = mask_encoding.decoder(pred_mask[b]).view(-1,28,28)
                # tmp_mask = tmp_mask[indices[b][0]]
                # tmp_mask2 = torch.full_like(tmp_mask,0).cuda()
                # tmp_mask2[tmp_mask>0.4]=1
                # masks_pred.append(tmp_mask2)
            else:
                # post_processing
                num_proposals = self.cfg.MODEL.SWINTS.TEST_NUM_PROPOSALS
                scores_per_image, topk_indices = scores.flatten(0, 1).topk(num_proposals, sorted=False)
                labels_per_image = labels[topk_indices]
                box_pred_per_image = bboxes.view(-1, 1, 4).repeat(1, 2, 1).view(-1, 4)
                box_pred_per_image = box_pred_per_image[topk_indices]
                mask_pred_per_image = mask_logits.view(-1, self.cfg.MODEL.SWINTS.MASK_DIM)
                mask_pred_per_image = mask_encoding.decoder(mask_pred_per_image, is_train=False)
                mask_pred_per_image = mask_pred_per_image.view(-1, 1, 28, 28)
                n, c, w, h = mask_pred_per_image.size()
                mask_pred_per_image = torch.repeat_interleave(mask_pred_per_image,2,1).view(-1, c, w, h)
                mask_pred_per_image = mask_pred_per_image[topk_indices]
                proposal_features = proposal_features.view(-1, 1, self.hidden_dim).repeat(1, 2, 1).view(-1, self.hidden_dim)
                proposal_features = proposal_features[topk_indices]
                proposal_boxes_pred.append(Boxes(box_pred_per_image))
                gt_masks.append(mask_pred_per_image)
                inter_class_logits.append(scores_per_image)
                inter_pred_bboxes.append(box_pred_per_image)
                inter_pred_masks.append(mask_pred_per_image)
                inter_pred_label.append(labels_per_image)

        # get recognition roi region
        if targets:
            rec_map = self.box_pooler_rec([features[0]], gt_boxes)
            # pred_roi_features = self.box_pooler_rec([features[0]], proposal_boxes_pred)
            # masks_pred = torch.cat(masks_pred).cuda()
            gt_masks = torch.cat(gt_masks).cuda()
            # rec_map = torch.cat((gt_roi_features,pred_roi_features),0)
            # gt_masks = torch.cat((gt_masks,masks_pred),0)
        else:
            rec_map = self.box_pooler_rec([features[0]], proposal_boxes_pred)
            # rec_map = self.box_pooler_rec(features, proposal_boxes_pred)
            gt_masks = torch.cat(gt_masks).cuda()
            nr_boxes = rec_map.shape[0]
        if targets:
            rec_map = rec_map[:self.cfg.MODEL.REC_HEAD.BATCH_SIZE]
        else:
            gt_masks_b = torch.full_like(gt_masks,0).cuda()
            gt_masks_b[gt_masks>0.4]=1
            gt_masks_b = gt_masks_b.squeeze()
            gt_masks = gt_masks_b
            del gt_masks_b
        if targets:
            return proposal_features, gt_masks[:self.cfg.MODEL.REC_HEAD.BATCH_SIZE], idx, rec_map, target_rec[:self.cfg.MODEL.REC_HEAD.BATCH_SIZE]
        else:
            return inter_class_logits, inter_pred_bboxes, inter_pred_masks, inter_pred_label, proposal_features, gt_masks, idx, rec_map, nr_boxes

    def forward(self, features, init_bboxes, init_features, targets = None, mask_encoding = None, matcher=None):
    
        inter_class_logits = []
        inter_pred_bboxes = []
        inter_pred_masks = []
        inter_pred_label = []
        inter_proposal_features = []
        bs = len(features[0])
        bboxes = init_bboxes
        proposal_features = init_features.clone()
        # for i_idx in range(len(features)):
        #    features[i_idx] = self.conv[i_idx](features[i_idx]) + features[i_idx]
        for i, rcnn_head in enumerate(self.head_series):

            class_logits, pred_bboxes, proposal_features, mask_logits = rcnn_head(features, bboxes, proposal_features, self.box_pooler)
            if self.return_intermediate:
                inter_class_logits.append(class_logits)
                inter_pred_bboxes.append(pred_bboxes)
                inter_pred_masks.append(mask_logits)
                inter_proposal_features.append(proposal_features)
            bboxes = pred_bboxes.detach()
        
        # extract recognition feature.
        N, nr_boxes = bboxes.shape[:2]
        if targets:
            proposal_features, gt_masks, idx, rec_map, target_rec = \
                self.extra_rec_feat(matcher, mask_encoding, targets, N, bboxes, class_logits, pred_bboxes, mask_logits, proposal_features, features)
        else:
            inter_class_logits, inter_pred_bboxes, inter_pred_masks, inter_pred_label, proposal_features, gt_masks, idx, rec_map, nr_boxes = \
                self.extra_rec_feat(matcher, mask_encoding, targets, N, bboxes, [class_logits], [pred_bboxes], [mask_logits], [proposal_features], features)
        # rec_map = self.cnn(rec_map)
        rec_proposal_features = proposal_features.clone()
        if targets:
            # if rec_map.shape[0]==0:
            #     rec_map = torch.zeros((1, 256, 28, 56)).to(features[0].device)
            #     target_rec = torch.zeros((1, 25)).to(features[0].device).long()
            #     rec_proposal_features = torch.zeros((1, 1, 256)).to(features[0].device)
            #     gt_masks =  torch.zeros((1, 28, 28)).to(features[0].device)
            #     N = 1
            #     idx = None
            #     nr_boxes = 1
            rec_result, decoded_scores = self.rec_stage(rec_map, rec_proposal_features, gt_masks, N, nr_boxes, idx, target_rec)
        else:
            rec_result, decoded_scores = self.rec_stage(rec_map, rec_proposal_features, gt_masks, N, nr_boxes)
            rec_result = torch.tensor(rec_result)
            decoded_scores = torch.tensor(decoded_scores)
        if self.return_intermediate:
                return torch.stack(inter_class_logits), torch.stack(inter_pred_bboxes), torch.stack(inter_pred_masks), rec_result, decoded_scores
        return class_logits[None], pred_bboxes[None], mask_logits[None]


class RCNNHead(nn.Module):

    def __init__(self, cfg, d_model, num_classes, dim_feedforward=2048, nhead=8, dropout=0.1, activation="relu",
                 scale_clamp: float = _DEFAULT_SCALE_CLAMP, bbox_weights=(2.0, 2.0, 1.0, 1.0)):
        super().__init__()

        self.d_model = d_model

        # dynamic.
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.inst_interact = DynamicConv(cfg)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ELU(inplace=True)

        # cls.
        num_cls = cfg.MODEL.SWINTS.NUM_CLS
        cls_module = list()
        for _ in range(num_cls):
            cls_module.append(nn.Linear(d_model, d_model, False))
            cls_module.append(nn.LayerNorm(d_model))
            cls_module.append(nn.ELU(inplace=True))
        self.cls_module = nn.ModuleList(cls_module)

        # reg.
        num_reg = cfg.MODEL.SWINTS.NUM_REG
        reg_module = list()
        for _ in range(num_reg):
            reg_module.append(nn.Linear(d_model, d_model, False))
            reg_module.append(nn.LayerNorm(d_model))
            reg_module.append(nn.ELU(inplace=True))
        self.reg_module = nn.ModuleList(reg_module)

        # mask.
        num_mask = cfg.MODEL.SWINTS.NUM_MASK
        mask_module = list()
        for _ in range(num_mask):
            mask_module.append(nn.Linear(d_model, d_model, False))
            mask_module.append(nn.LayerNorm(d_model))
            mask_module.append(nn.ELU(inplace=True))
        self.mask_module = nn.ModuleList(mask_module)
        self.mask_logits = nn.Linear(d_model, cfg.MODEL.SWINTS.MASK_DIM)

        # pred.
        self.class_logits = nn.Linear(d_model, num_classes)
        self.bboxes_delta = nn.Linear(d_model, 4)
        self.scale_clamp = scale_clamp
        self.bbox_weights = bbox_weights


    def forward(self, features, bboxes, pro_features, pooler):
        """
        :param bboxes: (N, nr_boxes, 4)
        :param pro_features: (N, nr_boxes, d_model)
        """

        N, nr_boxes = bboxes.shape[:2]
        # roi_feature.
        proposal_boxes = list()
        for b in range(N):
            proposal_boxes.append(Boxes(bboxes[b]))
        roi_features = pooler(features, proposal_boxes)
        roi_features = roi_features.view(N * nr_boxes, self.d_model, -1).permute(2, 0, 1)        

        # self_att.
        pro_features = pro_features.view(N, nr_boxes, self.d_model).permute(1, 0, 2)
        pro_features2 = self.self_attn(pro_features, pro_features, value=pro_features)[0]
        pro_features = pro_features + self.dropout1(pro_features2)

        del pro_features2

        pro_features = self.norm1(pro_features)

        # inst_interact.
        pro_features = pro_features.view(nr_boxes, N, self.d_model).permute(1, 0, 2).reshape(1, N * nr_boxes, self.d_model)
        pro_features2 = self.inst_interact(pro_features, roi_features)
        pro_features = pro_features + self.dropout2(pro_features2)

        del pro_features2

        obj_features = self.norm2(pro_features)

        # obj_feature.
        obj_features2 = self.linear2(self.dropout(self.activation(self.linear1(obj_features))))
        obj_features = obj_features + self.dropout3(obj_features2)

        del obj_features2

        obj_features = self.norm3(obj_features)
        
        fc_feature = obj_features.transpose(0, 1).reshape(N * nr_boxes, -1)
        cls_feature = fc_feature.clone()
        reg_feature = fc_feature.clone()

        mask_feature = fc_feature.clone()

        del fc_feature

        for mask_layer in self.mask_module:
            mask_feature = mask_layer(mask_feature)
        mask_logits = self.mask_logits(mask_feature)
        del mask_feature

        for cls_layer in self.cls_module:
            cls_feature = cls_layer(cls_feature)
        for reg_layer in self.reg_module:
            reg_feature = reg_layer(reg_feature)
        class_logits = self.class_logits(cls_feature)
        bboxes_deltas = self.bboxes_delta(reg_feature)

        del cls_feature
        del reg_feature

        pred_bboxes = self.apply_deltas(bboxes_deltas, bboxes.view(-1, 4))
        
        return class_logits.view(N, nr_boxes, -1), pred_bboxes.view(N, nr_boxes, -1), obj_features, mask_logits.view(N, nr_boxes, -1)
    

    def apply_deltas(self, deltas, boxes):
        """
        Apply transformation `deltas` (dx, dy, dw, dh) to `boxes`.

        Args:
            deltas (Tensor): transformation deltas of shape (N, k*4), where k >= 1.
                deltas[i] represents k potentially different class-specific
                box transformations for the single box boxes[i].
            boxes (Tensor): boxes to transform, of shape (N, 4)
        """
        boxes = boxes.to(deltas.dtype)

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.bbox_weights
        dx = deltas[:, 0::4] / wx
        dy = deltas[:, 1::4] / wy
        dw = deltas[:, 2::4] / ww
        dh = deltas[:, 3::4] / wh

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.scale_clamp)
        dh = torch.clamp(dh, max=self.scale_clamp)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w  # x1
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h  # y1
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w  # x2
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h  # y2

        return pred_boxes


class DynamicConv(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.hidden_dim = cfg.MODEL.SWINTS.HIDDEN_DIM
        self.dim_dynamic = cfg.MODEL.SWINTS.DIM_DYNAMIC
        self.num_dynamic = cfg.MODEL.SWINTS.NUM_DYNAMIC
        self.num_params = self.hidden_dim * self.dim_dynamic
        self.dynamic_layer = nn.Linear(self.hidden_dim, self.num_dynamic * self.num_params)

        self.norm1 = nn.LayerNorm(self.dim_dynamic)
        self.norm2 = nn.LayerNorm(self.hidden_dim)

        self.activation = nn.ELU(inplace=True)

        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        num_output = self.hidden_dim * pooler_resolution ** 2
        self.out_layer = nn.Linear(num_output, self.hidden_dim)
        self.norm3 = nn.LayerNorm(self.hidden_dim)

    def forward(self, pro_features, roi_features):
        '''
        pro_features: (1,  N * nr_boxes, self.d_model)
        roi_features: (49, N * nr_boxes, self.d_model)
        '''
        features = roi_features.permute(1, 0, 2)
        parameters = self.dynamic_layer(pro_features).permute(1, 0, 2)

        param1 = parameters[:, :, :self.num_params].view(-1, self.hidden_dim, self.dim_dynamic)
        param2 = parameters[:, :, self.num_params:].view(-1, self.dim_dynamic, self.hidden_dim)

        del parameters

        features = torch.bmm(features, param1)

        del param1

        features = self.norm1(features)
        features = self.activation(features)

        features = torch.bmm(features, param2)

        del param2

        features = self.norm2(features)
        features = self.activation(features)

        features = features.flatten(1)
        features = self.out_layer(features)
        features = self.norm3(features)
        features = self.activation(features)

        return features

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])



class Block_down(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.down_conv = nn.Conv2d(dim, dim, 3, 2, 1) # depthwise conv

    def forward(self, x):
        x = self.down_conv(x)
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        x = input + self.drop_path(x)
        return x

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        x = input + self.drop_path(x)
        return x

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ChannelAttention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        bs, c, h, w = x.shape
        x = x.flatten(2,3).transpose(1,2)
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        k = k * self.scale
        attention = k.transpose(-1, -2) @ v
        attention = attention.softmax(dim=-1)
        x = (attention @ q.transpose(-1, -2)).transpose(-1, -2)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = x.transpose(1,2).reshape(bs, c, h, w)
        return x
