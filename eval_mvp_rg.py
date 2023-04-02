import argparse
import copy
import glob
import numpy as np
import os
import pdb
import torch
import open3d as o3d
from easydict import EasyDict as edict
from tqdm import tqdm

from data import MVP_RG, get_dataloader
from models import architectures, GCNet, vote
from utils import decode_config, npy2pcd, pcd2npy, execute_global_registration, \
    npy2feat, vis_plys, setup_seed, fmat, to_tensor, get_blue, get_yellow
from metrics import Error_R, Error_t, RMSE

CUR = os.path.dirname(os.path.abspath(__file__))


def main(args):
    setup_seed(22) 
    config = decode_config(os.path.join(CUR, 'configs', 'mvp_rg.yaml'))
    config = edict(config)
    config.architecture = architectures[config.dataset]
    config.num_workers = 4
    test_dataset = MVP_RG(root=args.data_root, 
                          split='val',
                          rot_mag=config.rot_mag,
                          trans_mag=config.trans_mag,
                          overlap_radius=config.overlap_radius)

    test_dataloader, neighborhood_limits = get_dataloader(config=config,
                                                          dataset=test_dataset,
                                                          batch_size=config.batch_size,
                                                          num_workers=config.num_workers,
                                                          shuffle=False,
                                                          neighborhood_limits=None)

    print(neighborhood_limits)
    model = GCNet(config)
    use_cuda = not args.no_cuda
    if use_cuda:
        model = model.cuda()
        model.load_state_dict(torch.load(args.checkpoint))
    else:
        model.load_state_dict(
            torch.load(args.checkpoint, map_location=torch.device('cpu')))
    model.eval()

    Ts, gt_Ts, srcs = [], [], []
    with torch.no_grad():
        for pair_ind, inputs in enumerate(tqdm(test_dataloader)):
            if use_cuda:
                for k, v in inputs.items():
                    if isinstance(v, list):
                        for i in range(len(v)):
                            inputs[k][i] = inputs[k][i].cuda()
                    else:
                        inputs[k] = inputs[k].cuda()

            batched_feats_h, batched_feats_m, batched_feats_l = model(inputs)
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
            srcs.append(source_npy)

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
            source_saliency_scores = feats_src_h[:, -1].detach().cpu().numpy()
            target_saliency_scores = feats_tgt_h[:, -1].detach().cpu().numpy()

            source_scores = source_overlap_scores * source_saliency_scores
            target_scores = target_overlap_scores * target_saliency_scores
            
            npoints = args.npts
            if source_npy.shape[0] > npoints:
                p = source_scores / np.sum(source_scores)
                idx = np.random.choice(len(source_npy), size=npoints, replace=False, p=p)
                source_npy = source_npy[idx]
                source_feats_h = source_feats_h[idx]
                source_feats_m = source_feats_m[idx]
                source_feats_l = source_feats_l[idx]
            
            # if target_npy.shape[0] > npoints:
            #     p = target_scores / np.sum(target_scores)
            #     idx = np.random.choice(len(target_npy), size=npoints, replace=False, p=p)
            #     target_npy = target_npy[idx]
            #     target_feats_h = target_feats_h[idx]
            #     target_feats_m = target_feats_m[idx]
            #     target_feats_l = target_feats_l[idx]

            after_vote = vote(source_npy=source_npy, 
                              target_npy=target_npy, 
                              source_feats=[source_feats_h, source_feats_m, source_feats_l], 
                              target_feats=[target_feats_h, target_feats_m, target_feats_l], 
                              voxel_size=config.first_subsampling_dl,
                              use_cuda=use_cuda)
            source_npy, target_npy, source_feats_npy, target_feats_npy = after_vote
            
            M = torch.cdist(to_tensor(source_feats_npy, use_cuda), to_tensor(target_feats_npy, use_cuda))
            row_max_inds = torch.min(M, dim=-1)[1].cpu().numpy()
            col_max_inds = torch.min(M, dim=0)[1].cpu().numpy()

            source, target = npy2pcd(source_npy), npy2pcd(target_npy)

            source_feats, target_feats = npy2feat(source_feats_npy), npy2feat(target_feats_npy)
            pred_T, estimate = execute_global_registration(source=source,
                                                           target=target,
                                                           source_feats=source_feats,
                                                           target_feats=target_feats,
                                                           voxel_size=0.02)
            Ts.append(pred_T)
            gt_Ts.append(T)

            if args.vis:
                source_ply = npy2pcd(source_npy_raw)
                source_ply.paint_uniform_color(get_yellow())
                estimate_ply = copy.deepcopy(source_ply).transform(pred_T)
                target_ply = npy2pcd(target_npy_raw)
                target_ply.paint_uniform_color(get_blue())
                vis_plys([target_ply, estimate_ply], need_color=False)
   
    Ts, gt_Ts, srcs = np.array(Ts), np.array(gt_Ts), np.array(srcs)
    
    rot_error = Error_R(Ts[:, :3, :3], gt_Ts[:, :3, :3])
    trans_error =  Error_t(Ts[:, :3, 3], gt_Ts[:, :3, 3])
    rmse = RMSE(srcs, Ts[:, :3, :3], gt_Ts[:, :3, :3], Ts[:, :3, 3], gt_Ts[:, :3, 3])
    re = np.mean(rot_error)
    te = np.mean(trans_error)
    rmse = np.mean(rmse)

    print('Rotation error: ', fmat(re))
    print('translation error: ', fmat(te))
    print('RMSE: ', fmat(rmse))
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--npts', type=int, default=768,
                        help='the number of sampled points for registration')
    parser.add_argument('--data_root', required=True, help='data root')
    parser.add_argument('--checkpoint', required=True, help='checkpoint path')
    parser.add_argument('--vis', action='store_true',
                        help='whether for visualization')
    parser.add_argument('--no_cuda', action='store_true',
                        help='whether to use cuda')
    args = parser.parse_args()
    main(args)
