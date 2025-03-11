import warnings

import torch
import numpy as np
import pytorch3d.transforms as pytorch3d_t

from ICP_Flow.utils_icp import apply_icp
from ICP_Flow.utils_hist import estimate_init_pose
from ICP_Flow.utils_check import sanity_check, check_transformation
from ICP_Flow.utils_helper import transform_points_batch, nearest_neighbor_batch, setdiff1d, match_segments_descend, pad_segment

warnings.filterwarnings('ignore')

def match_pcds(args, src_points, dst_points, src_labels, dst_labels):

    src_labels_unq = torch.unique(src_labels, return_counts=False).long()
    dst_labels_unq = torch.unique(dst_labels, return_counts=False).long()
    labels_unq = torch.unique(torch.cat([src_labels_unq, dst_labels_unq], axis=0), return_counts=False)

    # # # stage 1: match static: overlapped clusters
    pairs = torch.stack([labels_unq, labels_unq], dim=1)
    mask = pairs.min(dim=1)[0]>=0 # remove ground
    pairs = pairs[mask]
    pairs_true = sanity_check(args, src_points, dst_points, src_labels, dst_labels, pairs)

    if len(pairs_true)>0: 
        pairs_sta, transformations_sta = match_pairs(args, src_points, dst_points, src_labels, dst_labels, pairs_true) 
    else:
        pairs_sta, transformations_sta = torch.tensor([]).reshape(0, 10), torch.tensor([]).reshape(0, 4, 4)

    # stage 2: match dynamic
    # # # remove matched near-static pairs:
    if len(pairs_sta)<len(labels_unq):
        if len(pairs_sta)>0:
            src_labels_unq = setdiff1d(src_labels_unq, pairs_sta[:, 0])
            dst_labels_unq = setdiff1d(dst_labels_unq, pairs_sta[:, 1])

        pairs = torch.stack([src_labels_unq.repeat_interleave(len(dst_labels_unq)), dst_labels_unq.repeat(len(src_labels_unq))], dim=1)
        pairs_true = sanity_check(args, src_points, dst_points, src_labels, dst_labels, pairs)
    else:
        pairs_true = torch.zeros(0, 2)

    if len(pairs_true)>0: 
        pairs_dyn, transformations_dyn = match_pairs(args, src_points, dst_points, src_labels, dst_labels, pairs_true) 
    else:
        pairs_dyn, transformations_dyn = torch.tensor([]).reshape(0, 10), torch.tensor([]).reshape(0, 4, 4)
    
    pairs_matched = torch.cat([pairs_sta, pairs_dyn], dim=0)
    transformations_matched = torch.cat([transformations_sta, transformations_dyn], dim=0)
    # assert len(pairs_matched)>0 # likely to be bugs or outliers

    return pairs_matched, transformations_matched


def match_pairs(args, src_points, dst_points, src_labels, dst_labels, pairs):
    src_labels_unq = torch.unique(src_labels)
    dst_labels_unq = torch.unique(dst_labels)
    matrix_errors = torch.zeros((len(src_labels_unq), len(dst_labels_unq), 2)) + 1e8
    matrix_inliers = torch.zeros((len(src_labels_unq), len(dst_labels_unq), 2)) + 0.0
    matrix_ratios = torch.zeros((len(src_labels_unq), len(dst_labels_unq), 2)) + 0.0
    matrix_ious = torch.zeros((len(src_labels_unq), len(dst_labels_unq), 2)) + 0.0
    matrix_transformations = torch.zeros((len(src_labels_unq), len(dst_labels_unq), 4, 4)) 

    assert len(pairs)>0
    segs_src = []
    segs_dst = []
    for pair in pairs:
        src = src_points[src_labels==pair[0], 0:3]
        dst = dst_points[dst_labels==pair[1], 0:3]
        src = pad_segment(src, args.max_points)
        dst = pad_segment(dst, args.max_points)
        # always match the smaller one to the larger one
        segs_src.append(src)
        segs_dst.append(dst)

    segs_src = torch.stack(segs_src, dim=0)
    segs_dst = torch.stack(segs_dst, dim=0)
    transformations = hist_icp(args, segs_src, segs_dst)
    errors, inliers, ratios, ious, translations, rotations = match_eval(args, segs_src, segs_dst, transformations)
    # reject unreliable matches
    num_matches = 0
    for k, (pair, error, inlier, ratio, iou, translation, rotation, transformation) in \
        enumerate(zip(pairs, errors, inliers, ratios, ious, translations, rotations, transformations)):
        if not check_transformation(args, translation, rotation, min(iou)):
            continue
        src_idx = torch.nonzero(src_labels_unq == pair[0])
        dst_idx = torch.nonzero(dst_labels_unq == pair[1])
        matrix_errors[src_idx, dst_idx, :] = error
        matrix_inliers[src_idx, dst_idx, :] = inlier
        matrix_ratios[src_idx, dst_idx, :] = ratio
        matrix_ious[src_idx, dst_idx, :] = iou
        matrix_transformations[src_idx, dst_idx] = transformation
        num_matches += 1

    if num_matches>0:
        matrix_errors_min, _ = matrix_errors.min(-1)
        src_idxs, dst_idxs = match_segments_descend(matrix_errors_min)
        valid = matrix_errors_min[src_idxs, dst_idxs] < args.thres_error
        src_idxs = src_idxs[valid]
        dst_idxs = dst_idxs[valid]

        pairs = torch.cat([src_labels_unq[src_idxs][:, None], dst_labels_unq[dst_idxs][:, None], 
                                matrix_errors[src_idxs, dst_idxs], 
                                matrix_inliers[src_idxs, dst_idxs], 
                                matrix_ratios[src_idxs, dst_idxs], 
                                matrix_ious[src_idxs, dst_idxs]], 
                                axis=1)

        transformations = matrix_transformations[src_idxs, dst_idxs]

    else:
        pairs = torch.tensor([]).reshape(0, 2+matrix_errors.shape[-1]+matrix_inliers.shape[-1]+matrix_ratios.shape[-1]+matrix_ious.shape[-1])
        transformations = torch.tensor([]).reshape(0, 4, 4)

    return pairs, transformations

def hist_icp(args, src, dst):
    mask1 = src[:, :, -1]>0.0
    mask2 = dst[:, :, -1]>0.0
    # alwyas match the smaller one to the larger one
    idxs = mask1.sum(dim=1)>mask2.sum(dim=1)
    src_ = src.clone()
    dst_ = dst.clone()
    src_[idxs] = dst[idxs]
    dst_[idxs] = src[idxs]

    with torch.no_grad():
        init_poses_ = estimate_init_pose(args, src_, dst_) 
        transformations_ = apply_icp(args, src_, dst_, init_poses_)

    if sum(idxs)>0:
        transformations = transformations_.clone()
        transformations[idxs] = torch.linalg.inv(transformations_[idxs])
    else:
        transformations = transformations_
    return transformations

def match_eval(args, pcd1, pcd2, transformations):
    pcd1_tmp = transform_points_batch(pcd1, transformations)
    src = pcd1_tmp
    dst = pcd2
    src_mask = pcd1[:, :, -1]>0.0
    dst_mask = pcd2[:, :, -1]>0.0
    _, src_error = nearest_neighbor_batch(src, dst)
    _, dst_error = nearest_neighbor_batch(dst, src)

    src_inlier = torch.logical_and(src_error < args.thres_dist, src_mask).float()
    dst_inlier = torch.logical_and(dst_error < args.thres_dist, dst_mask).float()

    src_ratio = torch.sum(src_inlier, dim=1) / torch.sum(src_mask, dim=1)
    dst_ratio = torch.sum(dst_inlier, dim=1) / torch.sum(dst_mask, dim=1)

    src_iou = torch.sum(src_inlier, dim=1) / (torch.sum(src_mask, dim=1) + torch.sum(dst_mask, dim=1) - torch.sum(dst_inlier, dim=1))
    dst_iou = torch.sum(dst_inlier, dim=1) / (torch.sum(src_mask, dim=1) + torch.sum(dst_mask, dim=1) - torch.sum(src_inlier, dim=1))

    src_error = (src_error * src_mask).sum(1) / src_mask.sum(1)
    dst_error = (dst_error * dst_mask).sum(1) / dst_mask.sum(1)

    src_mean = (src[:, :, 0:3] * src_mask[:, :, None]).sum(dim=1) / src_mask.sum(dim=1, keepdim=True)
    src_ori_mean = (pcd1[:, :, 0:3] * src_mask[:, :, None]).sum(dim=1) / src_mask.sum(dim=1, keepdim=True)

    translations = src_mean - src_ori_mean
    rotations = pytorch3d_t.matrix_to_euler_angles(transformations[:, 0:3, 0:3], convention='ZYX') * 180./np.pi

    return torch.stack([src_error, dst_error], dim=1), \
            torch.stack([src_inlier.sum(1), dst_inlier.sum(1)], dim=1), \
            torch.stack([src_ratio, dst_ratio], dim=1), \
            torch.stack([src_iou, dst_iou], dim=1), \
                translations, rotations
