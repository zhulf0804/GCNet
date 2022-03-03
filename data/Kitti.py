import copy
import glob
import numpy as np
import os
import pickle
import random
import torch
import open3d as o3d
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
CUR = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append(os.path.dirname(CUR))
from utils import npy2pcd, pcd2npy, vis_plys, get_correspondences, format_lines, voxel_ds, normal
import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling


class Kitti(Dataset):
    def __init__(self, root, split, aug, voxel_size, overlap_radius, max_coors, 
                 noise_scale=0.01,
                 augment_scale_min=0.8,
                 augment_scale_max=1.2,
                 augment_shift_range=2.0):
        super().__init__()
        self.root = root
        self.split = split
        self.aug = aug
        self.overlap_radius = overlap_radius
        self.voxel_size = voxel_size
        self.max_coors = max_coors
        self.noise_scale = noise_scale
        self.augment_scale_min = augment_scale_min
        self.augment_scale_max = augment_scale_max
        self.augment_shift_range = augment_shift_range
        
        self.dataset_root = os.path.join(root, 'dataset')
        self.icp_root = os.path.join(root, 'icp2')
        os.makedirs(self.icp_root, exist_ok=True)

        self.pose_cache, self.icp_cache = {}, {}
        
        assert split in ['train', 'val', 'test']
        seq_file = os.path.join(CUR, 'Kitti', f'{split}_kitti.txt')
        with open(seq_file, 'r') as f:
            lines = f.readlines()
            seq_ids = [f'{int(line.strip()):02}' for line in lines]
        self.pairs_info = self.prepare_pairs(seq_ids)

        if split=='test':
            self.pairs_info.remove(['08', 15, 58])
        print(f'Num_{split}: {len(self.pairs_info)}')

    def prepare_pairs(self, seq_ids):
        pairs_info = []
        for seq_id in seq_ids:
            pose_file = os.path.join(self.dataset_root, 'poses', f'{seq_id}.txt')
            poses = np.genfromtxt(pose_file) # (n, 12)
            poses = np.array([np.vstack([pose.reshape(3, 4), [0, 0, 0, 1]]) for pose in poses]) # (n, 4, 4)
            self.pose_cache[seq_id] = poses
            Ts = poses[:, :3, 3] # (n, 3)
            dists = np.sqrt(np.sum((Ts[:, None, :] - Ts[None, :, :]) ** 2, axis=-1)) # (n, n)
            
            bin_path = os.path.join(self.dataset_root, 'sequences', seq_id, 'velodyne')
            bin_files = sorted(glob.glob(os.path.join(bin_path, '*.bin')))
            bin_ids = [int(os.path.split(bin_file)[-1][:-4]) for bin_file in bin_files]
            
            # select pairs which are about 10m away from each other
            dists_bool = dists > 10
            cur_bin_id, mmax_id = bin_ids[0], np.max(bin_ids)
            while cur_bin_id < mmax_id:
                valid_ids = np.where(dists_bool[cur_bin_id][cur_bin_id:cur_bin_id + 100])[0]
                # print(dists_bool[cur_bin_id][cur_bin_id:cur_bin_id + 100])
                # print(cur_bin_id, valid_ids)
                if len(valid_ids) == 0:
                    cur_bin_id += 1
                    continue
                next_bin_id = cur_bin_id + valid_ids[0] - 1
                pairs_info.append([seq_id, cur_bin_id, next_bin_id])
                cur_bin_id = next_bin_id + 1
        return pairs_info

    @property
    def velo2cam(self):
        try:
            velo2cam = self._velo2cam
        except AttributeError:
            R = np.array([
                7.533745e-03, -9.999714e-01, -6.166020e-04, 1.480249e-02, 7.280733e-04,
                -9.998902e-01, 9.998621e-01, 7.523790e-03, 1.480755e-02
            ]).reshape(3, 3)
            t = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01]).reshape(3, 1)
            velo2cam = np.hstack([R, t])
            self._velo2cam = np.vstack((velo2cam, [0, 0, 0, 1])).T
        return self._velo2cam

    def __len__(self):
        return len(self.pairs_info)

    def __getitem__(self, item):
        seq_id, src_id, tgt_id = self.pairs_info[item]
        src_points_path = os.path.join(os.path.join(self.dataset_root, 'sequences', seq_id, 'velodyne', f'{src_id:06}.bin'))
        tgt_points_path = os.path.join(os.path.join(self.dataset_root, 'sequences', seq_id, 'velodyne', f'{tgt_id:06}.bin'))
        src_points = np.fromfile(src_points_path, dtype=np.float32).reshape(-1, 4)[:, :3]
        tgt_points = np.fromfile(tgt_points_path, dtype=np.float32).reshape(-1, 4)[:, :3]
        src_position, tgt_position = self.pose_cache[seq_id][[src_id, tgt_id]]

        # Following D3Feat and PREDATOR, we use ICP to refine the ground_truth pose, and we don't voxelize the point clouds here.
        key = f'{seq_id}_{src_id}_{tgt_id}'
        filename = os.path.join(self.icp_root, f'{key}.npy')
        if key not in self.icp_cache:
            if not os.path.exists(filename):
                # print('missing ICP files, recompute it')
                M = (self.velo2cam @ src_position.T @ np.linalg.inv(tgt_position.T)
                            @ np.linalg.inv(self.velo2cam)).T
                reg = o3d.registration.registration_icp(npy2pcd(src_points), npy2pcd(tgt_points), 0.2, M,
                                                        o3d.registration.TransformationEstimationPointToPoint(),
                                                        o3d.registration.ICPConvergenceCriteria(max_iteration=200))
                T = reg.transformation
                np.save(filename, T)
            else:
                T = np.load(filename)
            self.icp_cache[key] = T
        else:
            T = self.icp_cache[key]
        
        src_pcd, tgt_pcd = normal(npy2pcd(src_points), radius=3, max_nn=64, loc=(0, 0, 100)), normal(npy2pcd(tgt_points), radius=3, max_nn=64, loc=(0, 0, 100))
        src_normals = np.array(src_pcd.normals).astype(np.float32) 
        tgt_normals = np.array(tgt_pcd.normals).astype(np.float32)
        len_src, len_tgt = len(src_points), len(tgt_points)
        new_points, new_len, new_normals = cpp_subsampling.subsample_batch(np.vstack([src_points, tgt_points]),
                                                                           [len_src, len_tgt],
                                                                           features=np.vstack([src_normals, tgt_normals]),
                                                                           sampleDl=self.voxel_size,
                                                                           max_p=0,
                                                                           verbose=0)
        src_points, tgt_points = new_points[:new_len[0]], new_points[new_len[0]:]
        src_normals, tgt_normals = new_normals[:new_len[0]], new_normals[new_len[0]:]

        # src_points = pcd2npy(voxel_ds(npy2pcd(src_points), self.voxel_size))
        # tgt_points = pcd2npy(voxel_ds(npy2pcd(tgt_points), self.voxel_size))

        coors = get_correspondences(npy2pcd(src_points),
                                    npy2pcd(tgt_points),
                                    T,
                                    self.overlap_radius)

        if coors.shape[0] < self.max_coors and self.split == 'train':
            return self.__getitem__(np.random.choice(len(self.pairs_info), 1)[0])

        src_feats = np.ones_like(src_points[:, :1], dtype=np.float32)
        tgt_feats = np.ones_like(tgt_points[:, :1], dtype=np.float32)

        src_points_raw = copy.deepcopy(src_points)
        tgt_points_raw = copy.deepcopy(tgt_points)

        if self.aug:
            # noise
            src_points += (np.random.rand(src_points.shape[0], 3) - 0.5) * self.noise_scale
            tgt_points += (np.random.rand(tgt_points.shape[0], 3) - 0.5) * self.noise_scale

            # rotation
            euler_ab = np.random.rand(3) * 2 * np.pi
            rot_ab = Rotation.from_euler('zyx', euler_ab).as_matrix()
            if np.random.rand() > 0.5:
                src_points = src_points @ rot_ab.T
                src_normals = src_normals @ rot_ab.T
            else:
                tgt_points = tgt_points @ rot_ab.T
                tgt_normals = tgt_normals @ rot_ab.T

            # scale
            scale = self.augment_scale_min + (self.augment_scale_max - self.augment_scale_min) * random.random()
            src_points = src_points * scale
            tgt_points = tgt_points * scale

            # shift
            shift_src = np.random.uniform(-self.augment_shift_range, self.augment_shift_range, 3)
            shift_tgt = np.random.uniform(-self.augment_shift_range, self.augment_shift_range, 3)
            src_points = src_points + shift_src
            tgt_points = tgt_points + shift_tgt

        pair = dict(
            src_points=src_points,
            tgt_points=tgt_points,
            src_feats=src_feats,
            tgt_feats=tgt_feats,
            src_normals=src_normals,
            tgt_normals=tgt_normals,
            transf=T,
            coors=coors,
            src_points_raw=src_points_raw,
            tgt_points_raw=tgt_points_raw)
        return pair
