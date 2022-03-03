import os, h5py
import numpy as np
import open3d as o3d
from torch.utils.data import Dataset
from utils import npy2pcd, get_correspondences, normal


def jitter_pcd(pcd, sigma=0.01, clip=0.05):
    pcd += np.clip(sigma * np.random.randn(*pcd.shape), -1 * clip, clip)
    return pcd


def random_pose(max_angle, max_trans):
    R = random_rotation(max_angle)
    t = random_translation(max_trans)
    return np.concatenate([np.concatenate([R, t], 1), [[0, 0, 0, 1]]], 0)


def random_rotation(max_angle):
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    angle = np.random.rand() * max_angle
    A = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle) * A + (1 - np.cos(angle)) * np.dot(A, A)
    return R


def random_translation(max_dist):
    t = np.random.randn(3)
    t /= np.linalg.norm(t)
    t *= np.random.rand() * max_dist
    return np.expand_dims(t, 1)


class MVP_RG(Dataset):
    def __init__(self, root, split, rot_mag, trans_mag, overlap_radius):
        """
        Args:
            root (str): Folder containing processed dataset
            subset (str): Dataset subset, either 'train' or 'test'
            categories (list): Categories to use
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self._root = root
        self._subset = split
        self.max_angle, self.max_trans = rot_mag, trans_mag
        self.overlap_radius = overlap_radius  # 0.04

        if split == "train":
            h5_filelist = [os.path.join(root, 'MVP_Train_RG.h5')]
        elif split == "val":
            h5_filelist = [os.path.join(root, 'MVP_Test_RG.h5')]
        elif split == "test":
            h5_filelist = [os.path.join(root, 'MVP_ExtraTest_RG.h5')]

        self._src_tol, self._tgt_tol, self._transforms_tol, self._labels = \
            self._read_h5_files(h5_filelist)
        # self._data, self._labels = self._data[:32], self._labels[:32, ...]

    def __getitem__(self, item):
        sample = {'points_src': self._src_tol[item, :, :],
                  'points_ref': self._tgt_tol[item, :, :],
                  'transform_gt': self._transforms_tol[item, :, :],
                  'label': self._labels[item],
                  'idx': np.array(item, dtype=np.int32)}

        if 'train' in self._subset:
            transform = random_pose(self.max_angle, self.max_trans / 2)
            pose1 = random_pose(np.pi, self.max_trans)
            pose2 = transform @ pose1
            sample['transform_gt'] = transform
            sample['points_src'][:, :3] = sample['points_src'][:, :3] @ pose1[:3, :3].T + pose1[:3, 3]
            sample['points_ref'][:, :3] = sample['points_ref'][:, :3] @ pose2[:3, :3].T + pose2[:3, 3]

        # transform to our format
        src_points = sample['points_src'][:, :3].astype(np.float32)
        tgt_points = sample['points_ref'][:, :3].astype(np.float32)
        rot = sample['transform_gt'][:3, :3].astype(np.float32)
        trans = sample['transform_gt'][:3, 3][:, None].astype(np.float32)
        transf = np.eye(4, dtype=np.float32)
        transf[:3, :3] = rot
        transf[:3, 3:] = trans
        matching_inds = get_correspondences(npy2pcd(src_points),
                                            npy2pcd(tgt_points),
                                            transf,
                                            self.overlap_radius)

        src_pcd = normal(npy2pcd(src_points), radius=0.2, max_nn=30) #, loc=(0, 0, 1)) # new loc
        tgt_pcd = normal(npy2pcd(tgt_points), radius=0.2, max_nn=30) #, loc=(0, 0, 1)) # new loc
        src_normals = np.array(src_pcd.normals).astype(np.float32) 
        tgt_normals = np.array(tgt_pcd.normals).astype(np.float32)

        src_feats = np.ones_like(src_points[:, :1]).astype(np.float32)
        tgt_feats = np.ones_like(tgt_points[:, :1]).astype(np.float32)
        
        pair = dict(
            src_points=src_points,
            tgt_points=tgt_points,
            src_feats=src_feats,
            tgt_feats=tgt_feats,
            src_normals=src_normals,
            tgt_normals=tgt_normals,
            transf=transf,
            coors=matching_inds,
            src_points_raw=src_points,
            tgt_points_raw=tgt_points)
        return pair

    def __len__(self):
        return self._labels.shape[0]

    @property
    def classes(self):
        return self._classes

    def _read_h5_files(self, fnames):

        src_tol, tgt_tol, transforms_tol = [], [], []
        all_labels = []

        for fname in fnames:
            f = h5py.File(fname, mode='r')
            labels = f['cat_labels'][:].astype('int32')
            if self._subset == "test":
                src = np.array(f['rotated_src'][:].astype('float32'))
                tgt = np.array(f['rotated_tgt'][:].astype('float32'))
                transforms = np.array([np.eye(4, dtype=np.float32) for _ in range(src.shape[0])]).astype(np.float32)
            else:
                if "train" in self._subset:
                    src = np.array(f['src'][:].astype('float32'))
                    tgt = np.array(f['tgt'][:].astype('float32'))
                    transforms = np.array([np.eye(4, dtype=np.float32) for _ in range(src.shape[0])]).astype(np.float32)
                elif self._subset == "val":
                    src = np.array(f['rotated_src'][:].astype('float32'))
                    tgt = np.array(f['rotated_tgt'][:].astype('float32'))
                    transforms = np.array(
                        f['transforms'][:].astype('float32'))

            src_tol.append(src)
            tgt_tol.append(tgt)
            transforms_tol.append(transforms)
            all_labels.append(labels)
        src_tol = np.concatenate(src_tol, axis=0).astype(np.float32)
        tgt_tol = np.concatenate(tgt_tol, axis=0).astype(np.float32)
        transforms_tol = np.concatenate(transforms_tol, axis=0).astype(np.float32)
        all_labels = np.concatenate(all_labels, axis=0).astype(np.float32)
        return src_tol, tgt_tol, transforms_tol, all_labels
