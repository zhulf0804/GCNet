import numpy as np
import torch
import torch.nn as nn
from utils import square_dists


class Loss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.final_feats_dim = config.final_feats_dim
        self.n_samples = config.max_points
        self.pos_margin = config.pos_margin
        self.neg_margin = config.neg_margin
        self.pos_radius = config.pos_radius
        self.safe_radius = config.safe_radius
        self.matchability_radius = config.matchability_radius

        self.log_scale = config.log_scale
        self.w_circle_loss = config.w_circle_loss
        self.w_overlap_loss = config.w_overlap_loss
        # self.w_saliency_loss = config.w_saliency_loss

        self.soft_plus = nn.Softplus()
        self.bce_loss = nn.BCELoss(reduction='none')

    def circle_loss(self, coords_dist, feats_dist):
        '''

        :param coors_dist: (n, m)
        :param feats_dist: (n, m)
        :return:
        '''
        # set weights.
        # for positive pairs, if feats_dist < self.pos_margin, set weight=0,
        # else set weight=feats_dist - self.pos_margin
        pos_mask = coords_dist < self.pos_radius
        neg_mask = coords_dist > self.safe_radius
        pos_weight = pos_mask * torch.clamp(feats_dist - self.pos_margin, min=0)
        neg_weight = neg_mask * torch.clamp(self.neg_margin - feats_dist, min=0)
        pos_weight.detach_()
        neg_weight.detach_()

        pos_loss_row = torch.logsumexp(self.log_scale * pos_weight * (feats_dist - self.pos_margin), dim=1)
        neg_loss_row = torch.logsumexp(self.log_scale * neg_weight * (self.neg_margin - feats_dist), dim=1)
        pos_loss_col = torch.logsumexp(self.log_scale * pos_weight * (feats_dist - self.pos_margin), dim=0)
        neg_loss_col = torch.logsumexp(self.log_scale * neg_weight * (self.neg_margin - feats_dist), dim=0)

        loss_row = self.soft_plus(pos_loss_row + neg_loss_row) / self.log_scale
        loss_col = self.soft_plus(pos_loss_col + neg_loss_col) / self.log_scale
        rol_sel = (pos_mask.sum(1) > 0) * (neg_mask.sum(1) > 0)
        col_sel = (pos_mask.sum(0)) > 0 * (neg_mask.sum(0) > 0)
        return (loss_row[rol_sel].mean() + loss_col[col_sel].mean()) / 2

    def overlap_loss(self, ol_scores, ol_gt):
        '''

        :param ol_scores: (N, ) or (M, )
        :param ol_gt: (N, ) or (M, )
        :return:
        '''

        ol_loss = self.bce_loss(ol_scores, ol_gt)
        ratio = ol_gt.sum(0) / ol_gt.size(0)
        weights = torch.zeros_like(ol_gt).to(ol_gt)
        weights[ol_gt > 0.5] = 1 - ratio
        weights[ol_gt < 0.5] = ratio
        weighted_ol_loss = torch.mean(ol_loss * weights)
        return weighted_ol_loss
    
    def saliency_loss(self, saliency, saliency_gt):
        '''

        :param saliency: (n, ) or (m, )
        :param saliency_gt: (n, ) or (m, )
        :return:
        '''
        sa_loss = self.bce_loss(saliency, saliency_gt)
        ratio = saliency_gt.sum(0) / saliency_gt.size(0)
        weights = torch.zeros_like(saliency_gt).to(saliency_gt)
        weights[saliency_gt > 0.5] = 1 - ratio
        weights[saliency_gt < 0.5] = ratio
        weighted_sa_loss = torch.mean(sa_loss * weights)
        return weighted_sa_loss

    def get_recall(self, coords_dist, feats_dist):
        '''
        used for updating saliency loss weight and selecting the best recall checkpoint.
        :param coors_dist: (n, m)
        :param feats_dist: (n, m)
        '''
        pos_mask = coords_dist < self.pos_radius
        n_gt = torch.sum(torch.sum(pos_mask, dim=1) > 0) + 1e-8
        inds = torch.min(feats_dist, dim=1)[1]
        sel_dist = torch.gather(coords_dist, dim=-1, index=inds[:, None])[pos_mask.sum(1) > 0]
        n_pred = torch.sum(sel_dist < self.pos_radius)
        recall = n_pred / n_gt
        return recall

    def forward(self, coords_src, coords_tgt, feats_src, feats_tgt, feats_src_m, feats_tgt_m, feats_src_l, feats_tgt_l, coors, transf, w_saliency):
        '''

        :param coords_src: (N, 3)
        :param coords_tgt: (M, 3)
        :param feats_src: (N, C)
        :param feats_tgt: (M, C)
        :param coors: (L, 2)
        :param transf: (4, 4)
        :return:
        '''
        loss_dict = {}
        # 0. output parsing
        ol_scores_src, ol_scores_tgt = feats_src[:, -2], feats_tgt[:, -2]
        saliency_src, saliency_tgt = feats_src[:, -1], feats_tgt[:, -1]
        feats_src, feats_tgt = feats_src[:, :-2], feats_tgt[:, :-2]

        R, t = transf[:3, :3], transf[:3, 3:]
        coords_src = (R @ coords_src.transpose(0, 1) + t).transpose(0, 1)

        # there are lots of repetitive indices, such as (5, 10), (5, 11), (5, 13) ...
        # we need to implement deduplication operations. important for memory!
        # tolist() is a good function.
        inds_src, inds_tgt = list(set(coors[:, 0].int().tolist())), \
                             list(set(coors[:, 1].int().tolist()))

        # 1. overlap loss
        ol_gt_src = torch.zeros(coords_src.size(0), dtype=torch.float32).to(coords_src)
        ol_gt_tgt = torch.zeros(coords_tgt.size(0), dtype=torch.float32).to(coords_tgt)
        ol_gt_src[inds_src] = 1
        ol_gt_tgt[inds_tgt] = 1
        overlap_loss_v = 0.5 * self.overlap_loss(ol_scores_src, ol_gt_src) + \
                         0.5 * self.overlap_loss(ol_scores_tgt, ol_gt_tgt)
        loss_dict['overlap_loss'] = overlap_loss_v

        # 2. saliency loss (based overlapping points)
        coords_src_sel, coords_tgt_sel = coords_src[inds_src], coords_tgt[inds_tgt]
        feats_src_sel, feats_tgt_sel = feats_src[inds_src], feats_tgt[inds_tgt]
        feats_dist = torch.matmul(feats_src_sel, feats_tgt_sel.transpose(0, 1)) # (n, m)
        inds1 = torch.max(feats_dist, dim=1)[1]
        dists1 = torch.norm(coords_src_sel - coords_tgt_sel[inds1], dim=1) # (n, )
        inds2 = torch.max(feats_dist, dim=0)[1]
        dists2 = torch.norm(coords_tgt_sel - coords_src_sel[inds2], dim=1) # (m, )
        saliency_src_sel = saliency_src[inds_src]
        saliency_tgt_sel = saliency_tgt[inds_tgt]
        saliency_loss_v = 0.5 * self.saliency_loss(saliency_src_sel, (dists1 < self.matchability_radius).float()) + \
                          0.5 * self.saliency_loss(saliency_tgt_sel, (dists2 < self.matchability_radius).float())
        loss_dict['saliency_loss'] = saliency_loss_v

        # 3. descriptor loss
        # select n_samples (which matches each other) points for feature description loss
        # based on overlapping points, we filter some points.
        # because the values of overlap_radius and pos_radius may be different.
        inds = torch.norm(coords_src[coors[:, 0]] - coords_tgt[coors[:, 1]], dim=1) < self.pos_radius - 0.001
        coors = coors[inds]

        if coors.size(0) > self.n_samples:
            inds = np.random.choice(coors.size(0), self.n_samples, replace=False)
            coors = coors[inds]

        # there may be some repeated points in source point cloud.
        # we need to keep valid pairs.
        inds_src, inds_tgt = coors[:, 0], coors[:, 1]
        coords_src, coords_tgt = coords_src[inds_src], coords_tgt[inds_tgt]
        feats_src, feats_tgt = feats_src[inds_src], feats_tgt[inds_tgt]
        feats_src_m, feats_tgt_m = feats_src_m[inds_src], feats_tgt_m[inds_tgt]
        feats_src_l, feats_tgt_l = feats_src_l[inds_src], feats_tgt_l[inds_tgt]
        coords_dist = torch.sqrt(square_dists(coords_src[None, :, :], coords_tgt[None, :, :]).squeeze(0))
        feats_dist = torch.sqrt(square_dists(feats_src[None, :, :], feats_tgt[None, :, :]).squeeze(0))
        feats_dist_m = torch.sqrt(square_dists(feats_src_m[None, :, :], feats_tgt_m[None, :, :]).squeeze(0))
        feats_dist_l = torch.sqrt(square_dists(feats_src_l[None, :, :], feats_tgt_l[None, :, :]).squeeze(0))


        circle_loss_v = self.circle_loss(coords_dist=coords_dist,
                                         feats_dist=feats_dist)
        circle_loss_vm = self.circle_loss(coords_dist=coords_dist,
                                          feats_dist=feats_dist_m)
        circle_loss_vl = self.circle_loss(coords_dist=coords_dist,
                                          feats_dist=feats_dist_l)
        recall = self.get_recall(coords_dist=coords_dist, feats_dist=feats_dist)
        recall_m = self.get_recall(coords_dist=coords_dist, feats_dist=feats_dist_m)
        recall_l = self.get_recall(coords_dist=coords_dist, feats_dist=feats_dist_l)
        loss_dict['circle_loss'] = circle_loss_v
        loss_dict['circle_loss_m'] = circle_loss_vm
        loss_dict['circle_loss_l'] = circle_loss_vl
        loss_dict['recall'] = recall
        loss_dict['recall_m'] = recall_m
        loss_dict['recall_l'] = recall_l

        loss = self.w_circle_loss * circle_loss_v + \
               self.w_circle_loss * circle_loss_vm + \
               self.w_circle_loss * circle_loss_vl + \
               self.w_overlap_loss * overlap_loss_v + \
               w_saliency * saliency_loss_v
        loss_dict['total_loss'] = loss

        return loss_dict
