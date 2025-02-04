#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sklearn

import time
import argparse

import torch
import numpy as np
import open3d as o3d

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from utils import visualize_pcd, get_clusters
from point_fitting import icp_transform, good_match


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--eps", type=float, default=0.7, help="DBSCAN epsilon")
    parser.add_argument(
        "--min_points", type=int, default=10, help="DBSCAN minimum points"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load reference vehicles
    ref_veh = o3d.io.read_point_cloud("exports/cloud_test.ply")
    ref_veh_points = np.asarray(ref_veh.points)
    ref_veh_colors = np.asarray(ref_veh.colors)
    ref_veh1 = ref_veh_points[np.all(ref_veh_colors == [1, 0, 0], axis=1)]
    ref_veh2 = ref_veh_points[np.all(ref_veh_colors == [0, 1, 0], axis=1)]

    # Load and preprocess the point cloud
    data = np.load("exports/scene-0158.npy", allow_pickle=True).T
    labels = get_clusters(data, args.eps, args.min_points)
    cluster_num = len(np.unique(labels))
    # print(f"Number of clusters: {cluster_num}")

    # Do ICP
    cluster1 = data[labels == 0]
    cluster2 = data[labels == 1]

    cluster_cent1 = (np.min(cluster1, axis=0) + np.max(cluster1, axis=0)) / 2
    cluster_cent2 = (np.min(cluster2, axis=0) + np.max(cluster2, axis=0)) / 2

    cluster_cent1 = np.mean(cluster1, axis=0)
    cluster_cent2 = np.mean(cluster2, axis=0)

    # Plot the 3D points and bounding box
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    points = cluster1 - cluster_cent1

    # Plot points
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c="red",
        marker="o",
        label="Points",
        s=5,
    )

    points = cluster2 - cluster_cent2

    # Plot points
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c="blue",
        marker="o",
        label="Points",
        s=5,
    )

    # Labels and legend
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Points and Bounding Box")
    ax.legend()

    # plt.show()
    # exit()

    start = time.time()
    res = icp_transform(cluster1, cluster2)
    print(f"Time taken: {time.time() - start:.2f} s")
    good = good_match(
        cluster1, cluster2, res, save_name="exp/cluster_test.ply", visualize=True
    )
