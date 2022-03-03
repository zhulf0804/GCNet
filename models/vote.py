import numpy as np
import torch
from utils import to_tensor


def get_coor_points(source_feats_npy, target_feats_npy, target_npy, use_cuda):
    dists = torch.cdist(to_tensor(source_feats_npy, use_cuda), to_tensor(target_feats_npy, use_cuda)) # (n, m)
    inds = torch.min(dists, dim=-1)[1].cpu().numpy()
    return target_npy[inds], inds


def vote(source_npy, target_npy, source_feats, target_feats, voxel_size, use_cuda=True):
    source_feats_h, source_feats_m, source_feats_l = source_feats
    target_feats_h, target_feats_m, target_feats_l = target_feats

    coor_y1, coor_inds1 = get_coor_points(source_feats_h, target_feats_h, target_npy, use_cuda)
    coor_y2, coor_inds2 = get_coor_points(source_feats_m, target_feats_m, target_npy, use_cuda)
    coor_y3, coor_inds3 = get_coor_points(source_feats_l, target_feats_l, target_npy, use_cuda)

    d12 = np.sqrt(np.sum((coor_y1 - coor_y2) ** 2, axis=-1))
    d13 = np.sqrt(np.sum((coor_y1 - coor_y3) ** 2, axis=-1))
    d23 = np.sqrt(np.sum((coor_y2 - coor_y3) ** 2, axis=-1))

    thresh = voxel_size * 2
    source_sel_h = np.any([d12 < thresh, d13 < thresh], axis=0)
    source_sel_m = d23 < thresh
    source_sel = np.any([source_sel_h, source_sel_m], axis=0)

    source_sel_replace = np.all([~source_sel_h, source_sel_m], axis=0)
    source_feats_h[source_sel_replace] = source_feats_m[source_sel_replace]
    target_feats_h[coor_inds2[source_sel_replace]] = target_feats_m[coor_inds2[source_sel_replace]]

    source_npy = source_npy[source_sel]
    source_feats_h = source_feats_h[source_sel]

    after_vote =  [source_npy, target_npy, source_feats_h, target_feats_h]
    return after_vote
  