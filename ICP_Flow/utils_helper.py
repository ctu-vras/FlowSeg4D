import warnings

import torch
import numpy as np
import pytorch3d.ops as p3d
from scipy.spatial.transform import Rotation
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")


def nearest_neighbor_batch(src, dst):
    assert src.dim() == 3
    assert dst.dim() == 3
    assert len(src) == len(dst)
    b, num, _ = src.shape
    assert src.shape[2] >= 3
    assert dst.shape[2] >= 3
    result = p3d.knn_points(src[:, :, 0:3], dst[:, :, 0:3], K=1)
    dst_idxs = result.idx
    distances = result.dists
    return dst_idxs.view(b, num), distances.view(b, num).sqrt()


def nearest_neighbor(src, dst):
    """
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
    """
    neigh = NearestNeighbors(n_neighbors=1, p=2)
    neigh.fit(dst)
    distances, dst_idxs = neigh.kneighbors(src, return_distance=True)
    dst_idxs = dst_idxs.ravel()
    distances = distances.ravel()
    return dst_idxs, distances


def trackers2labels(label_i, label_j, pairs):
    unq_i = np.unique(label_i)
    unq_j = np.unique(label_j)
    # re-label tracked clusters to label_i, label_j
    # ground points: -1e8; untracked: -1; tracked: >=0
    label_i_track = label_i.copy()
    label_j_track = label_j.copy()
    label_i_track[label_i >= 0] = -1
    label_j_track[label_j >= 0] = -1

    for k, pair in enumerate(pairs):
        assert pair[0] in unq_i
        assert pair[1] in unq_j
        idx_i = label_i == pair[0]
        label_i_track[idx_i] = k
        # # let op:
        # enforcing one-to-one correspondence
        # idx_j = label_j==pair[1]
        # label_j_track[idx_j]=k
        # # if not one-to-one correspondence: find the first occurence
        idx_j = label_j == pair[1]
        idxs = np.flatnonzero(pairs[:, 1] == pair[1])
        label_j_track[idx_j] = idxs[0]

    return label_i_track, label_j_track


def transform_points_batch(xyz, pose):
    assert xyz.dim() == 3
    assert pose.dim() == 3
    assert xyz.shape[2] == 4
    assert pose.shape[1] == 4
    assert pose.shape[2] == 4
    assert len(xyz) == len(pose)
    b, n, _ = xyz.shape
    xyzh = torch.cat([xyz[:, :, 0:3], xyz.new_ones((b, n, 1))], dim=-1)
    xyzh_tmp = torch.bmm(xyzh, pose.permute(0, 2, 1))
    return torch.cat([xyzh_tmp[:, :, 0:3], xyz[:, :, -1:]], dim=-1)


def transform_points(xyz, pose):
    assert xyz.shape[1] == 3
    xyzh = np.concatenate([xyz, np.ones((len(xyz), 1))], axis=1)
    xyzh_tmp = xyzh @ pose.T
    return xyzh_tmp[:, 0:3]


def measure_R_t(src, dst):
    src_mean = src.mean(0)
    dst_mean = dst.mean(0)
    t = dst_mean - src_mean
    src_tmp = src - src_mean[None, :]
    dst_tmp = dst - dst_mean[None, :]
    H = src_tmp.T @ dst_tmp
    U, _, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    rot = Rotation.from_matrix(R)
    rot_angles = rot.as_euler("zyx", degrees=True)
    return rot_angles, t


def match_segments_descend(matrix_metric):
    src_idxs = torch.arange(0, len(matrix_metric))
    dst_idxs = torch.argmin(matrix_metric, dim=1)

    return src_idxs, dst_idxs


def match_segments_ascend(matrix_metric):
    src_idxs = torch.arange(0, len(matrix_metric))
    dst_idxs = torch.argmax(matrix_metric, dim=1)

    return src_idxs, dst_idxs


def get_bbox_tensor(points):
    x = torch.abs(points[:, 0].max() - points[:, 0].min())
    y = torch.abs(points[:, 1].max() - points[:, 1].min())
    z = torch.abs(points[:, 2].max() - points[:, 2].min())
    return sorted([x, y, z])


def setdiff1d(t1, t2):
    # assuming t2 is a subset of t1
    t1_unique = torch.unique(t1)
    t2_unique = torch.unique(t2)
    assert len(t1_unique) >= len(t2_unique)
    t12, counts = torch.cat([t1_unique, t2_unique]).unique(return_counts=True)
    diff = t12[torch.where(counts.eq(1))]
    return diff


def pad_segment(seg, max_points):
    padding = seg.new_zeros((max_points, 1)) + 1.0
    if len(seg) > max_points:
        sample_idxs = random_choice(len(seg), max_points)
        seg = seg[sample_idxs, :]
    elif len(seg) < max_points:
        padding[len(seg) :] = 0.0
        seg = torch.cat([seg, seg.new_zeros((max_points - len(seg), 3)) + 1e8], dim=0)
    else:
        pass
    assert len(seg) == max_points
    return torch.cat([seg, padding], axis=1)


def random_choice(m, n):
    assert m >= n
    perm = torch.randperm(m)
    return perm[0:n]

