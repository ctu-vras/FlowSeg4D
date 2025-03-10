#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sklearn

import time
import argparse

import torch
import numpy as np
import open3d as o3d

from utils import visualize_pcd, get_clusters
from point_fitting import icp_transform, good_match, icp_match


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--eps", type=float, default=0.7, help="DBSCAN epsilon")
    parser.add_argument(
        "--min_points", type=int, default=10, help="DBSCAN minimum points"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load and preprocess the point cloud
    data = np.load("exports/scene-0158.npy", allow_pickle=True).T
    labels = get_clusters(data, args.eps, args.min_points)
    cluster_num = len(np.unique(labels))

    # Do ICP
    cluster1 = data[labels == 0]
    cluster2 = data[labels == 1]

    # Save the clusters
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cluster1)
    o3d.io.write_point_cloud("exp/cluster1.ply", pcd)
    pcd.points = o3d.utility.Vector3dVector(cluster2)
    o3d.io.write_point_cloud("exp/cluster2.ply", pcd)

    start = time.time()
    res = icp_transform(cluster1, cluster2)
    # print(res)

    # source_cent = (np.min(cluster1, axis=0) + np.max(cluster1, axis=0)) / 2
    # cluster1 = torch.tensor(cluster1 - source_cent)
    # sample_cent = (np.min(cluster2, axis=0) + np.max(cluster2, axis=0)) / 2
    # cluster2 = torch.tensor(cluster2 - sample_cent)
    #
    # cdist = torch.cdist(cluster1, cluster2)
    # argmin = torch.argmin(cdist, dim=1)
    # cluster2 = cluster2[argmin]
    #
    # H = cluster2.T @ cluster1
    # U, S, V = torch.svd(H)
    # R = V.T @ U.T
    #
    # if torch.det(R) < 0:
    #     V[:, -1] *= -1
    #     R = V.T @ U.T
    #
    # t = sample_cent - R.numpy() @ source_cent
    #
    # res = icp_match(cluster1, cluster2, t[0], t[1], R)
    print(f"Time taken: {time.time() - start:.2f} s")
    # vis_cluster = torch.cat((torch.tensor(cluster2), res), dim=0)
    # visualize_pcd(
    #     vis_cluster,
    #     labels=np.concatenate((np.ones(len(cluster2)), 2 * np.ones(len(cluster1)))),
    # )
    good = good_match(
        cluster1,
        cluster2,
        res,
        save_name="exp/cluster_test.ply",
        visualize=True,
        verbose=True,
    )
