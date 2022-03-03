import os
import numpy as np
import nibabel.quaternions as nq

from metrics.kitti import Error_R, Error_t


def inlier_ratio_core(points_src, points_tgt, row_max_inds, col_max_inds, transf, inlier_threshold=0.1):

    R, t = transf[:3, :3], transf[:3, 3]

    points_src = points_src @ R.T + t
    inlier_mask = np.sum((points_src - points_tgt[row_max_inds]) ** 2, axis=1) < inlier_threshold ** 2
    inlier_ratio = np.sum(inlier_mask) / len(inlier_mask)

    # mutual inlier ratio
    mutual_corrs = []
    for i in range(len(points_src)):
        if col_max_inds[row_max_inds[i]] == i:
            mutual_corrs.append([i, row_max_inds[i]])
    mutual_corrs = np.array(mutual_corrs, dtype=np.int)
    mutual_mask = np.sum((points_src[mutual_corrs[:, 0]] - points_tgt[mutual_corrs[:, 1]]) ** 2, axis=1) < inlier_threshold ** 2
    mutual_inlier_ratio = np.sum(mutual_mask) / len(mutual_corrs)

    return inlier_ratio, mutual_inlier_ratio


def registration_recall_core(points_src, points_tgt, gt_corrs, pred_T):
    '''

    :param points_src: (n, 3)
    :param points_tgt: (m, 3)
    :param gt_corrs: (n1, 2)
    :param pred_T: (4, 4)
    :return: float
    '''
    points_src = points_src[gt_corrs[:, 0]]
    points_tgt = points_tgt[gt_corrs[:, 1]]
    R, t = pred_T[:3, :3], pred_T[:3, 3]
    points_src = points_src @ R.T + t
    mse = np.mean(np.sum((points_src - points_tgt) ** 2, axis=1))
    rmse = np.sqrt(mse)

    return rmse


class Metric(object):

    def __init__(self):
        self.err2 = 0.2 ** 2
        self.re_thre = 15
        self.te_thre = 30

    def benchmark(self, est_folder, gt_folder='configs/benchmarks/3DMatch'):
        # est_folder = 'snapshot/indoor/est_traj/3DMatch_1000_prob'
        # gt_folder = 'configs/benchmarks/3DMatch'
        scenes = sorted(os.listdir(gt_folder))

        n_valids, n_totals = [], []
        predator_style_recall_per_scene, dsc_style_recall_per_scene = [], []
        error_rs, error_ts = [], []
        for scene_i, scene in enumerate(scenes):
            est_pairs, est_traj = self.read_trajectory(os.path.join(est_folder, scene, 'est.log'))
            gt_pairs, gt_traj = self.read_trajectory(os.path.join(gt_folder, scene, 'gt.log'))
            n_fragments, gt_traj_cov = self.read_trajectory_info(os.path.join(gt_folder, scene, "gt.info"))

            n_valid = 0
            for ele in gt_pairs:
                diff = abs(int(ele[0]) - int(ele[1]))
                n_valid += diff > 1
            n_valids.append(n_valid)
            n_totals.append(len(est_traj))

            good_1, good_2, valid_num, error_r, error_t = self.evaluate_both_recall(est_pairs, gt_pairs, est_traj, gt_traj, n_fragments, gt_traj_cov)
            assert valid_num == n_valids[scene_i]
            predator_style_recall_per_scene.append(good_1 / valid_num)
            dsc_style_recall_per_scene.append(good_2 / len(est_pairs))
            error_rs.append(error_r)
            error_ts.append(error_t)

        return np.array(predator_style_recall_per_scene), \
               np.array(error_rs), \
               np.array(error_ts), \
               np.array(dsc_style_recall_per_scene), \
               np.array(n_valids), \
               np.array(n_totals)

    def evaluate_both_recall(self, est_pairs, gt_pairs, est_traj, gt_traj, n_fragments, gt_traj_cov):
        assert (gt_pairs == est_pairs).all()

        good_predator, good_dsc = 0, 0
        valid = 0
        flags = []
        for i in range(len(est_pairs)):
            ind_i, ind_j, _ = est_pairs[i]
            this_gt, this_pred, this_info = gt_traj[i], est_traj[i], gt_traj_cov[i]
            if self.dsc_style_recall(this_pred, this_gt):
                good_dsc += 1
            if int(ind_j) - int(ind_i) > 1:
                valid += 1
                if self.predator_style_recall(this_pred, this_gt, this_info):
                    good_predator += 1
                    flags.append(0)
                else:
                    flags.append(1)
            else:
                flags.append(2)
        error_rs = Error_R(est_traj[:, :3, :3], gt_traj[:, :3, :3])[np.array(flags) == 0]
        error_ts = Error_t(est_traj[:, :3, 3], gt_traj[:, :3, 3])[np.array(flags) == 0]
        error_r_mean, error_r_median = np.mean(error_rs), np.median(error_rs)
        error_t_mean, error_t_median = np.mean(error_ts), np.median(error_ts)

        return good_predator, good_dsc, valid, [error_r_mean, error_r_median], [error_t_mean, error_t_median]

    def dsc_style_recall(self, pred, gt):
        pred_R, pred_t = self.decompose_trans(pred)
        gt_R, gt_t = self.decompose_trans(gt)
        re = np.arccos(np.clip((np.trace(pred_R.T @ gt_R) - 1) / 2.0, -1, 1))
        te = np.sqrt(np.sum((pred_t - gt_t) ** 2))
        re = re * 180 / np.pi
        te = te * 100
        return bool(re < self.re_thre and te < self.te_thre)

    def predator_style_recall(self, pred, gt, info):

        p = self.computeTransformationErr(np.linalg.inv(gt) @ pred, info)
        return p <= self.err2

    def computeTransformationErr(self, trans, info):

        t = trans[:3, 3]
        r = trans[:3, :3]
        q = nq.mat2quat(r)
        er = np.concatenate([t, q[1:]], axis=0)
        p = er.reshape(1, 6) @ info @ er.reshape(6, 1) / info[0, 0]
        return p.item()

    def decompose_trans(self, trans):

        if len(trans.shape) == 3:
            return trans[:, :3, :3], trans[:, :3, 3:4]
        else:
            return trans[:3, :3], trans[:3, 3:4]

    def read_trajectory(self, filename, dim=4):
        with open(filename) as f:
            lines = f.readlines()

            # Extract the point cloud pairs
            keys = lines[0::(dim + 1)]
            temp_keys = []
            for i in range(len(keys)):
                temp_keys.append(keys[i].split('\t')[0:3])

            final_keys = []
            for i in range(len(temp_keys)):
                final_keys.append([temp_keys[i][0].strip(), temp_keys[i][1].strip(), temp_keys[i][2].strip()])

            traj = []
            for i in range(len(lines)):
                if i % 5 != 0:
                    traj.append(lines[i].split('\t')[0:dim])

            traj = np.asarray(traj, dtype=np.float).reshape(-1, dim, dim)
            final_keys = np.asarray(final_keys)

            return final_keys, traj

    def read_trajectory_info(self, filename, dim=6):

        with open(filename) as fid:
            contents = fid.readlines()
        n_pairs = len(contents) // 7
        assert (len(contents) == 7 * n_pairs)
        info_list = []
        n_frame = 0

        for i in range(n_pairs):
            frame_idx0, frame_idx1, n_frame = [int(item) for item in contents[i * 7].strip().split()]
            info_matrix = np.concatenate(
                [np.fromstring(item, sep='\t').reshape(1, -1) for item in contents[i * 7 + 1:i * 7 + 7]], axis=0)
            info_list.append(info_matrix)

        cov_matrix = np.asarray(info_list, dtype=np.float).reshape(-1, dim, dim)
        return n_frame, cov_matrix
