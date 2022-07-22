import copy
import numpy as np
import torch
import open3d as o3d
import random


def read_cloud(path, rt='pcd'):
    if path.endswith('.ply') or path.endswith('.pcd'):
        pcd = o3d.io.read_point_cloud(path)
    elif path.endswith('.pth'):
        points = torch.load(path)
        pcd = npy2pcd(points)
    else:
        raise NotImplementedError
    if rt == 'pcd':
        return pcd
    elif rt == 'npy':
        return np.asarray(pcd.points)
    else:
        raise NotImplementedError


def npy2pcd(npy):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(npy)
    return pcd


def pcd2npy(pcd):
    npy = np.array(pcd.points)
    return npy


def get_blue():
    """
    Get color blue for rendering
    """
    return [0, 0.651, 0.929]


def get_yellow():
    """
    Get color yellow for rendering
    """
    return [1, 0.706, 0]


def npy2feat(npy):
    feats = o3d.registration.Feature()
    feats.data = npy.T
    return feats


def normal(pcd, radius=0.1, max_nn=30, loc=(0, 0, 0)):
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn),
                         fast_normal_computation=False)
    pcd.orient_normals_towards_camera_location(loc)
    return pcd


def vis_plys(pcds, need_color=True):
    colors = [[1.0, 0, 0],
              [0, 1.0, 0],
              [0, 0, 1.0]]
    if need_color:
        for i, pcd in enumerate(pcds):
            color = colors[i] if i < 3 else [random.random() for _ in range(3)]
            pcd.paint_uniform_color(color)
    o3d.visualization.draw_geometries(pcds)


def format_lines(points, lines, colors=None):
    '''
    :param points: n x 3
    :param lines:  m x 2
    :param colors: m x 3
    :return:
    '''
    if colors is None:
        colors = [[1, 0, 0] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


# the speed is slow here ? we should test it later.
def get_correspondences(src_ply, tgt_ply, transf, search_radius, K=None):
    src_ply = copy.deepcopy(src_ply)
    src_ply.transform(transf)
    pcd_tree = o3d.geometry.KDTreeFlann(tgt_ply)
    src_npy = pcd2npy(src_ply)
    corrs = []
    for i in range(src_npy.shape[0]):
        point = src_npy[i]
        [k, idx, _] = pcd_tree.search_radius_vector_3d(point, search_radius)
        if K is not None:
            idx = idx[:K]
        for j in idx:
            corrs.append([i, j])
    return np.array(corrs)


def voxel_ds(ply, voxel_size):
    return ply.voxel_down_sample(voxel_size=voxel_size)


def batch_grid_subsampling(points, batches_len, sampleDl):
    '''

    :param points: shape=(n, 3), n = n1 + n2 + ... + nk, k denotes batch size
    :param batches_len: shape=(k, ), values = (n1, n2, ..., nk)
    :param sampleDl: float,
    :return:
    '''
    new_points, new_len = [], []
    st = 0
    points = points.cpu().numpy()
    batches_len = batches_len.cpu().numpy()
    for i in range(len(batches_len)):
        l = batches_len[i]
        cur_points = points[st:st+l]
        cur_ply = npy2pcd(cur_points)
        cur_ply_ds = voxel_ds(cur_ply, voxel_size=sampleDl)
        cur_points_ds = pcd2npy(cur_ply_ds)
        new_points.append(cur_points_ds)
        new_len.append(len(cur_points_ds))
        st += l
    return torch.from_numpy(np.vstack(new_points).astype(np.float32)), \
           torch.from_numpy(np.array(new_len, dtype=np.int32))


def batch_neighbors(batch_queries, batch_supports, q_batches, s_batches, radius, max_nn):
    '''
    # Open3d is too slow, so we won't use this method here.
    :param batch_queries: shape=(n, 3), n = n1 + n2 + ... + nk, k denotes batch size
    :param batch_supports: shape=(m, 3), m = m1 + m2 + ... + mk
    :param q_batches: shape=(k, ), values = (n1, n2, ..., nk)
    :param s_batches: shape=(k, ), values = (m1, m2, ..., mk)
    :param radius: float
    :return: shape=(n, max_nn)
    '''
    inds = []
    q_st, s_st = 0, 0
    for i in range(len(q_batches)):
        q_len, s_len = q_batches[i], s_batches[i]
        queries, supports = batch_queries[q_st:q_st+q_len], batch_supports[s_st:s_st+s_len]
        supports_ply = npy2pcd(supports)
        supports_tree = o3d.geometry.KDTreeFlann(supports_ply)
        for query_point in queries:
            [k, idx, _] = supports_tree.search_radius_vector_3d(query_point, radius)
            if k > max_nn:
                process_idx = list(idx)[:max_nn]
            else:
                process_idx = list(idx) + [len(batch_supports) - s_st] * (max_nn - k)
            inds.append(np.array(process_idx) + s_st)
        q_st += q_len
        s_st += s_len

    return np.array(inds).astype(np.int32)


def execute_global_registration(source, target, source_feats, target_feats, voxel_size):
    distance_threshold = voxel_size
    result = o3d.registration.registration_ransac_based_on_feature_matching(
        source, target, source_feats, target_feats, distance_threshold,
        o3d.registration.TransformationEstimationPointToPoint(False), 3, [
            o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.registration.RANSACConvergenceCriteria(50000, 1000))
    transformation = result.transformation
    estimate = copy.deepcopy(source)
    estimate.transform(transformation)
    return transformation, estimate
