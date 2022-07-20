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

from data import ThreeDMatch, get_dataloader
from models import architectures, NgeNet, vote
from utils import decode_config, npy2pcd, pcd2npy, execute_global_registration, \
                  npy2feat, vis_plys, setup_seed, fmat, to_tensor, get_blue, \
                  get_yellow
from metrics import inlier_ratio_core, registration_recall_core, Metric
from rich.console import Console
from rich.table import Table

CUR = os.path.dirname(os.path.abspath(__file__))


def get_scene_split(file_path):
    test_cats = ['7-scenes-redkitchen',
                 'sun3d-home_at-home_at_scan1_2013_jan_1',
                 'sun3d-home_md-home_md_scan9_2012_sep_30',
                 'sun3d-hotel_uc-scan3',
                 'sun3d-hotel_umd-maryland_hotel1',
                 'sun3d-hotel_umd-maryland_hotel3',
                 'sun3d-mit_76_studyroom-76-1studyroom2',
                 'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika']
    c = 0
    splits, ply_coors_ids, pairs_ids = [], [], []
    for cat in test_cats:
        with open(os.path.join(file_path, cat, 'gt.log'), 'r') as f:
            lines = f.readlines()
            stride = len(lines) // 5
            for line in lines[::5]:
                item = list(map(int, line.strip().split('\t')))
                ply_coors_ids.append(item)
        splits.append([c, c + stride])
        c += stride
    return splits, np.array(ply_coors_ids, dtype=np.int64), test_cats

def print_table(scenes, scene_recall, rre, rte):

    console = Console()
    table = Table(show_header=True, header_style="bold")

    columns = ["scene", "recall", "rre", "rte"]
    for col in columns:
        table.add_column(col)

    values = np.concatenate([scene_recall[:,None], rre[:, 1:], rte[:, 1:]], axis=1)

    for sid, vals in zip(scenes, values):
        table.add_row(sid, *[f'{v:.3f}' for v in vals])

    scene_recall_mean = np.mean(scene_recall)
    scene_recall_std = np.std(scene_recall)
    rre_mean = np.mean(rre[:, :1])
    rre_std = np.std(rre[:, 1])
    rte_mean = np.mean(rte[:, :1])
    rte_std = np.std(rte[:, 1])

    table.add_row('avg', *[f'{scene_recall_mean:.3f} +- {scene_recall_std:.3f}',
                           f'{rre_mean:.3f} +- {rre_std:.3f}',
                           f'{rte_mean:.3f} +- {rte_std:.3f}'])
    console.print(table)


def main(args):
    setup_seed(22)
    config = decode_config(os.path.join(CUR, 'configs', 'threedmatch.yaml'))
    config = edict(config)
    config.architecture = architectures[config.dataset]
    config.num_workers = 4
    test_dataset = ThreeDMatch(root=args.data_root,
                               split=args.benchmark,
                               aug=False,
                               overlap_radius=config.overlap_radius)

    test_dataloader, neighborhood_limits = get_dataloader(config=config,
                                                          dataset=test_dataset,
                                                          batch_size=config.batch_size,
                                                          num_workers=config.num_workers,
                                                          shuffle=False,
                                                          neighborhood_limits=None)

    print(neighborhood_limits)
    model = NgeNet(config)
    use_cuda = not args.no_cuda
    if use_cuda:
        model = model.cuda()
        model.load_state_dict(torch.load(args.checkpoint))
    else:
        model.load_state_dict(
            torch.load(args.checkpoint, map_location=torch.device('cpu')))
    model.eval()

    fmr_threshold = 0.05
    rmse_threshold = 0.2
    inlier_ratios, mutual_inlier_ratios = [], []
    mutual_feature_match_recalls, feature_match_recalls = [], []
    rmses, Ts = [], []
    metric = Metric()

    dist_thresh_maps = {
      '5000': config.first_subsampling_dl,
      '2500': config.first_subsampling_dl * 1.5,
      '1000': config.first_subsampling_dl * 1.5,
      '500': config.first_subsampling_dl * 1.5,
      '250': config.first_subsampling_dl * 2,
    }
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
                              voxel_size=config.first_subsampling_dl,
                              use_cuda=use_cuda)
            source_npy, target_npy, source_feats_npy, target_feats_npy = after_vote
            
            M = torch.cdist(to_tensor(source_feats_npy, use_cuda), to_tensor(target_feats_npy, use_cuda))
            row_max_inds = torch.min(M, dim=-1)[1].cpu().numpy()
            col_max_inds = torch.min(M, dim=0)[1].cpu().numpy()

            inlier_ratio, mutual_inlier_ratio = inlier_ratio_core(points_src=source_npy,
                                                                  points_tgt=target_npy,
                                                                  row_max_inds=row_max_inds,
                                                                  col_max_inds=col_max_inds,
                                                                  transf=transf.detach().cpu().numpy())

            inlier_ratios.append(inlier_ratio)
            mutual_inlier_ratios.append(mutual_inlier_ratio)
            feature_match_recalls.append(inlier_ratio > fmr_threshold)
            mutual_feature_match_recalls.append(mutual_inlier_ratio > fmr_threshold)


            source, target = npy2pcd(source_npy), npy2pcd(target_npy)


            source_feats, target_feats = npy2feat(source_feats_npy), npy2feat(target_feats_npy)
            pred_T, estimate = execute_global_registration(source=source,
                                                           target=target,
                                                           source_feats=source_feats,
                                                           target_feats=target_feats,
                                                           voxel_size=dist_thresh_maps[str(args.npts)])

            Ts.append(pred_T)

            coors_filter = {}
            for i, j in coors:
                if i not in coors_filter:
                    coors_filter[i] = j
            coors_filter = np.array([[i, j] for i, j in coors_filter.items()])
            rmse = registration_recall_core(points_src=source_npy_raw,
                                            points_tgt=target_npy_raw,
                                            gt_corrs=coors_filter,
                                            pred_T=pred_T)
            rmses.append(rmse)
            
            if args.vis:
                source_ply = npy2pcd(source_npy_raw)
                source_ply.paint_uniform_color(get_yellow())
                estimate_ply = copy.deepcopy(source_ply).transform(pred_T)
                target_ply = npy2pcd(target_npy_raw)
                target_ply.paint_uniform_color(get_blue())
                vis_plys([target_ply, estimate_ply], need_color=False)
    
    Ts = np.array(Ts)
    file_path = os.path.join(CUR, 'data', 'ThreeDMatch', 'gt', args.benchmark)
    splits, ply_coors_ids, scenes = get_scene_split(file_path=file_path)
    valid_idx = np.abs(ply_coors_ids[:, 0] - ply_coors_ids[:, 1]) > 1
    n_valids = []
    cat_inlier_ratios, cat_mutual_inlier_ratios = [], []
    cat_mutual_feature_match_recalls, cat_feature_match_recalls = [], []
    cat_registration_recalls = []
    for i, split in enumerate(splits):
        scene = scenes[i]
        cur_ply_coors_ids = ply_coors_ids[split[0]:split[1]]
        cur_saved_dir = os.path.join(args.saved_path, scene)
        os.makedirs(cur_saved_dir, exist_ok=True)
        cur_Ts = Ts[split[0]:split[1]]
        with open(os.path.join(cur_saved_dir, 'est.log'), 'w') as f:
            for idx in range(cur_Ts.shape[0]):
                p = cur_Ts[idx,:,:].tolist()
                f.write('\t'.join(map(str, cur_ply_coors_ids[idx])) + '\n')
                f.write('\n'.join('\t'.join(map('{0:.12f}'.format, p[i])) for i in range(4)))
                f.write('\n')

        m_inlier_ratio = np.mean(inlier_ratios[split[0]:split[1]])
        m_mutual_inlier_ratio = np.mean(mutual_inlier_ratios[split[0]:split[1]])
        m_feature_match_recall = np.mean(feature_match_recalls[split[0]:split[1]])
        m_mutual_feature_match_recall = np.mean(mutual_feature_match_recalls[split[0]:split[1]])

        valid_idx_split = valid_idx[split[0]:split[1]]
        n_valids.append(np.sum(valid_idx_split))
        cat_inlier_ratios.append(m_inlier_ratio)
        cat_mutual_inlier_ratios.append(m_mutual_inlier_ratio)
        cat_feature_match_recalls.append(m_feature_match_recall)
        cat_mutual_feature_match_recalls.append(m_mutual_feature_match_recall)

    print('=' * 20, f'Recall: {np.sum(n_valids)} pairs / {len(valid_idx)}', '=' * 20)
    scene_recall, error_r, error_t, pair_recall, n_valids, n_totals = metric.benchmark(est_folder=args.saved_path,
                        gt_folder=os.path.join(CUR, 'data', 'ThreeDMatch', 'gt', args.benchmark))

    print_table(scenes, scene_recall, error_r, error_t)
    print('Pair-level recall: ', fmat(np.sum(pair_recall * n_totals) / np.sum(n_totals)))

    print('=' * 20, 'IR and FMR', '=' * 20)
    print("Inlier ratio: ", fmat(np.mean(cat_inlier_ratios)))
    print("Mutual inlier ratio: ", fmat(np.mean(cat_mutual_inlier_ratios)))
    print("Feature match recall: ", fmat(np.mean(cat_feature_match_recalls)))
    print("Mutual feature match recall: ", fmat(np.mean(cat_mutual_feature_match_recalls)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--benchmark', default='3DMatch', help='3DMatch or 3DLoMatch')
    parser.add_argument('--data_root', required=True, help='data root')
    parser.add_argument('--checkpoint', required=True, help='checkpoint path')
    parser.add_argument('--saved_path', default='work_dirs', help='saved path')
    parser.add_argument('--npts', type=int, default=5000,
                        help='the number of sampled points for registration')
    parser.add_argument('--vis', action='store_true',
                        help='whether to visualize the point clouds')
    parser.add_argument('--no_cuda', action='store_true',
                        help='whether to use cuda')
    args = parser.parse_args()
    main(args)
