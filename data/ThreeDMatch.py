import numpy as np
import os
import pickle
import torch
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
CUR = os.path.dirname(os.path.abspath(__file__))
from utils import npy2pcd, pcd2npy, vis_plys, get_correspondences, format_lines, normal


class ThreeDMatch(Dataset):
    def __init__(self, root, split, aug, overlap_radius, noise_scale=0.005):
        super().__init__()
        self.root = root
        self.split = split
        self.aug = aug
        self.noise_scale = noise_scale
        self.overlap_radius = overlap_radius
        self.max_points = 30000
        
        pkl_path = os.path.join(CUR, 'ThreeDMatch', f'{split}_info.pkl')
        if not os.path.exists(pkl_path):
            pkl_path = os.path.join(CUR, 'ThreeDMatch', f'{split}.pkl')
        with open(pkl_path, 'rb') as f:
            self.infos = pickle.load(f)
        for k, v in self.infos.items():
            print(k, len(v), type(v))

    def __len__(self):
        return len(self.infos['rot'])

    def __getitem__(self, item):
        src_path, tgt_path = self.infos['src'][item], self.infos['tgt'][item] # str, str
        rot, trans = self.infos['rot'][item], self.infos['trans'][item] # (3, 3), (3, 1)
        overlap = self.infos['overlap'][item] # float

        src_points = torch.load(os.path.join(self.root, src_path)) # npy, (n, 3)
        tgt_points = torch.load(os.path.join(self.root, tgt_path)) # npy, (m, 3)

        # for gpu memory
        if (src_points.shape[0] > self.max_points):
            idx = np.random.permutation(src_points.shape[0])[:self.max_points]
            src_points = src_points[idx]
        if (tgt_points.shape[0] > self.max_points):
            idx = np.random.permutation(tgt_points.shape[0])[:self.max_points]
            tgt_points = tgt_points[idx]

        if self.aug:
            euler_ab = np.random.rand(3) * 2 * np.pi
            rot_ab = Rotation.from_euler('zyx', euler_ab).as_matrix()
            if np.random.rand() > 0.5:
                src_points = src_points @ rot_ab.T
                rot = rot @ rot_ab.T
            else:
                tgt_points = tgt_points @ rot_ab.T
                rot = rot_ab @ rot
                trans = rot_ab @ trans

            src_points += (np.random.rand(src_points.shape[0], 3) - 0.5) * self.noise_scale
            tgt_points += (np.random.rand(tgt_points.shape[0], 3) - 0.5) * self.noise_scale

        T = np.eye(4).astype(np.float32)
        T[:3, :3] = rot
        T[:3, 3:] = trans

        coors = get_correspondences(npy2pcd(src_points),
                                    npy2pcd(tgt_points),
                                    T,
                                    self.overlap_radius)

        src_feats = np.ones_like(src_points[:, :1], dtype=np.float32)
        tgt_feats = np.ones_like(tgt_points[:, :1], dtype=np.float32)

        src_pcd, tgt_pcd = normal(npy2pcd(src_points)), normal(npy2pcd(tgt_points))
        src_normals = np.array(src_pcd.normals).astype(np.float32) 
        tgt_normals = np.array(tgt_pcd.normals).astype(np.float32)

        pair = dict(
            src_points=src_points,
            tgt_points=tgt_points,
            src_feats=src_feats,
            tgt_feats=tgt_feats,
            src_normals=src_normals,
            tgt_normals=tgt_normals,
            transf=T,
            coors=coors,
            src_points_raw=src_points,
            tgt_points_raw=tgt_points)
        return pair
