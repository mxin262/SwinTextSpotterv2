import torch
import torch.nn.functional as F
from torch import nn
from fvcore.nn import sigmoid_focal_loss_jit

from .util import box_ops
from .util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
from .util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou

from scipy.optimize import linear_sum_assignment


class Rec_SetCriterion(nn.Module):
    def __init__(self):
        super().__init__()
        self.matcher = Rec_HungarianMatcher()

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def forward(self, rec, loc, targets):

        # Retrieve the matching between the outputs of the last layer and the targets
        rec = rec.flatten(2,3).permute(0,2,1)
        loc = loc.flatten(2,3).permute(0,2,1)
        indices = self.matcher(rec, loc, targets)
        src_inx = self._get_src_permutation_idx(indices)
        tgt_inx = self._get_tgt_permutation_idx(indices)
        tgt_pos = torch.arange(targets.shape[1], device=loc.device)
        tgt_pos = tgt_pos.unsqueeze(0).repeat(targets.shape[0],1)
        det_rec_losses  = F.cross_entropy(rec[src_inx], targets[tgt_inx])
        pos_losses = F.cross_entropy(loc[src_inx], tgt_pos[tgt_inx])
        return det_rec_losses, pos_losses

class Rec_HungarianMatcher(nn.Module):
    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, cost_mask: float = 1):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_mask = cost_mask
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, rec, loc, targets):
        rec = rec.sigmoid()
        loc = loc.sigmoid()
        bs, num_queries, c = rec.shape
        tgt_pos = torch.arange(targets.shape[1])
        tgt_pos = tgt_pos.unsqueeze(0).repeat(bs,1)
        # Final cost matrix
        rec = rec.flatten(0,1)
        loc = loc.flatten(0,1)
        tgt_ids = targets.flatten(0,1)
        tgt_pos = tgt_pos.flatten(0,1)

        alpha = 0.25
        gamma = 2
        neg_cost_class = (1 - alpha) * (rec ** gamma) * (-(1 - rec + 1e-8).log())
        pos_cost_class = alpha * ((1 - rec) ** gamma) * (-(rec + 1e-8).log())
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        neg_cost_pos = (1 - alpha) * (loc ** gamma) * (-(1 - loc + 1e-8).log())
        pos_cost_pos = alpha * ((1 - loc) ** gamma) * (-(loc + 1e-8).log())
        cost_pos = pos_cost_pos[:, tgt_pos] - neg_cost_pos[:, tgt_pos]
        C = 2*cost_class + cost_pos
        C = C.view(bs, num_queries, -1).cpu() 
        sizes = [v.shape[0] for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]