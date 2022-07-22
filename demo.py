import argparse
import copy
import numpy as np
import os
import torch
from easydict import EasyDict as edict
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data import collate_fn
from models import architectures, NgeNet, vote
from utils import decode_config, npy2pcd, pcd2npy, execute_global_registration, \
                  npy2feat, setup_seed, get_blue, get_yellow, voxel_ds, normal, \
                  read_cloud, vis_plys

CUR = os.path.dirname(os.path.abspath(__file__))


class NgeNet_pipeline():
    def __init__(self, ckpt_path, voxel_size=0.025, cuda=True):
        self.voxel_size = voxel_size
        self.cuda = cuda
        config = self.prepare_config()
        self.neighborhood_limits = [38, 36, 35, 38]
        model = NgeNet(config)
        if self.cuda:
            model = model.cuda()
            model.load_state_dict(torch.load(ckpt_path))
        else:
            model.load_state_dict(
                torch.load(ckpt_path, map_location=torch.device('cpu')))
        self.model = model
        self.config = config
        self.model.eval()
    
    def prepare_config(self):
        config = decode_config(os.path.join(CUR, 'configs', 'threedmatch.yaml'))
        config = edict(config)
        config.first_subsampling_dl = self.voxel_size
        config.architecture = architectures[config.dataset]
        return config

    def prepare_inputs(self, source, target, voxel_size):
        src_pcd_input = pcd2npy(voxel_ds(copy.deepcopy(source), voxel_size))
        tgt_pcd_input = pcd2npy(voxel_ds(copy.deepcopy(target), voxel_size))
        src_feats = np.ones_like(src_pcd_input[:, :1])
        tgt_feats = np.ones_like(tgt_pcd_input[:, :1])

        src_pcd = normal(npy2pcd(src_pcd_input), radius=4*voxel_size, max_nn=30, loc=(0, 0, 0))
        tgt_pcd = normal(npy2pcd(tgt_pcd_input), radius=4*voxel_size, max_nn=30, loc=(0, 0, 0))
        src_normals = np.array(src_pcd.normals).astype(np.float32) 
        tgt_normals = np.array(tgt_pcd.normals).astype(np.float32)

        T = np.eye(4)
        coors = np.array([[0, 0], [1, 1]])
        src_pcd = pcd2npy(source)
        tgt_pcd = pcd2npy(target)

        pair = dict(
            src_points=src_pcd_input,
            tgt_points=tgt_pcd_input,
            src_feats=src_feats,
            tgt_feats=tgt_feats,
            src_normals=src_normals,
            tgt_normals=tgt_normals,
            transf=T,
            coors=coors,
            src_points_raw=src_pcd,
            tgt_points_raw=tgt_pcd)
        
        dict_inputs = collate_fn([pair], self.config, self.neighborhood_limits)
        if self.cuda:
            for k, v in dict_inputs.items():
                    if isinstance(v, list):
                        for i in range(len(v)):
                            dict_inputs[k][i] = dict_inputs[k][i].cuda()
                    else:
                        dict_inputs[k] = dict_inputs[k].cuda()
        
        return dict_inputs

    def pipeline(self, source, target, npts=20000):
        voxel_size = self.voxel_size
        inputs = self.prepare_inputs(source, target, voxel_size)

        batched_feats_h, batched_feats_m, batched_feats_l = self.model(inputs)
        stack_points = inputs['points']
        stack_lengths = inputs['stacked_lengths']
        coords_src = stack_points[0][:stack_lengths[0][0]]
        coords_tgt = stack_points[0][stack_lengths[0][0]:]
        feats_src_h = batched_feats_h[:stack_lengths[0][0]]
        feats_tgt_h = batched_feats_h[stack_lengths[0][0]:]
        feats_src_m = batched_feats_m[:stack_lengths[0][0]]
        feats_tgt_m = batched_feats_m[stack_lengths[0][0]:]
        feats_src_l = batched_feats_l[:stack_lengths[0][0]]
        feats_tgt_l = batched_feats_l[stack_lengths[0][0]:]

        coors = inputs['coors'][0] # list, [coors1, coors2, ..], preparation for batchsize > 1
        transf = inputs['transf'][0] # (1, 4, 4), preparation for batchsize > 1

        coors = coors.detach().cpu().numpy()
        T = transf.detach().cpu().numpy()

        source_npy = coords_src.detach().cpu().numpy()
        target_npy = coords_tgt.detach().cpu().numpy()

        source_npy_raw = copy.deepcopy(source_npy)
        target_npy_raw = copy.deepcopy(target_npy)
        source_feats_h = feats_src_h[:, :-2].detach().cpu().numpy()
        target_feats_h = feats_tgt_h[:, :-2].detach().cpu().numpy()
        source_feats_m = feats_src_m.detach().cpu().numpy()
        target_feats_m = feats_tgt_m.detach().cpu().numpy()
        source_feats_l = feats_src_l.detach().cpu().numpy()
        target_feats_l = feats_tgt_l.detach().cpu().numpy() 

        source_overlap_scores = feats_src_h[:, -2].detach().cpu().numpy()
        target_overlap_scores = feats_tgt_h[:, -2].detach().cpu().numpy()
        source_scores = source_overlap_scores
        target_scores = target_overlap_scores

        npoints = npts
        if npoints > 0:
            if source_npy.shape[0] > npoints:
                p = source_scores / np.sum(source_scores)
                idx = np.random.choice(len(source_npy), size=npoints, replace=False, p=p)
                source_npy = source_npy[idx]
                source_feats_h = source_feats_h[idx]
                source_feats_m = source_feats_m[idx]
                source_feats_l = source_feats_l[idx]
            
            if target_npy.shape[0] > npoints:
                p = target_scores / np.sum(target_scores)
                idx = np.random.choice(len(target_npy), size=npoints, replace=False, p=p)
                target_npy = target_npy[idx]
                target_feats_h = target_feats_h[idx]
                target_feats_m = target_feats_m[idx]
                target_feats_l = target_feats_l[idx]
        after_vote = vote(source_npy=source_npy, 
                            target_npy=target_npy, 
                            source_feats=[source_feats_h, source_feats_m, source_feats_l], 
                            target_feats=[target_feats_h, target_feats_m, target_feats_l], 
                            voxel_size=voxel_size,
                            use_cuda=self.cuda)
        
        source_npy, target_npy, source_feats_npy, target_feats_npy = after_vote
        source, target = npy2pcd(source_npy), npy2pcd(target_npy)
        source_feats, target_feats = npy2feat(source_feats_npy), npy2feat(target_feats_npy)
        pred_T, estimate = execute_global_registration(source=source,
                                                        target=target,
                                                        source_feats=source_feats,
                                                        target_feats=target_feats,
                                                        voxel_size=voxel_size*2)
        torch.cuda.empty_cache()
        return pred_T, npy2pcd(copy.deepcopy(source_npy_raw)).transform(pred_T)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--src_path', required=True, help='source point cloud path')
    parser.add_argument('--tgt_path', required=True, help='target point cloud path')
    parser.add_argument('--checkpoint', required=True, help='checkpoint path')
    parser.add_argument('--voxel_size', type=float, default=0.025, help='voxel size')
    parser.add_argument('--npts', type=int, default=5000,
                        help='the number of sampled points for registration')
    parser.add_argument('--no_vis', action='store_true',
                        help='whether to visualize the point clouds')
    parser.add_argument('--no_cuda', action='store_true',
                        help='whether to use cuda')
    args = parser.parse_args()

    # input data 
    source, target = read_cloud(args.src_path), read_cloud(args.tgt_path)

    # loading model
    cuda = not args.no_cuda
    model = NgeNet_pipeline(
        ckpt_path=args.checkpoint, 
        voxel_size=args.voxel_size, 
        cuda=cuda)
    
    # registration
    T, estimate = model.pipeline(source, target, npts=args.npts)
    print('Estimated transformation matrix: ', T)

    # vis
    if not args.no_vis:
        source.paint_uniform_color(get_yellow())
        source.estimate_normals()
        target.paint_uniform_color(get_blue())
        target.estimate_normals()
        vis_plys([source, target], need_color=False)

        estimate.paint_uniform_color(get_yellow())
        estimate.estimate_normals()
        vis_plys([estimate, target], need_color=False)
