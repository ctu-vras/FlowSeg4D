
import numpy as np
import os
import sys
import open3d as o3d
from utils_visualization import visualize_pcd, visualize_pcd_multiple, visualize_pcd_plotly

# # # form SegContrast
def cluster_dbscan(args, points):
    o3d.utility.random.seed(0) # only in open3d>=0.16
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    # clusterize pcd points
    labels = pcd.cluster_dbscan(eps=args.epsilon, min_points=args.min_cluster_size)
    labels = np.array(labels)
    lbls, counts = np.unique(labels, return_counts=True)
    # print('max num points per segment: ', max(counts))
    cluster_info = np.array(list(zip(lbls[1:], counts[1:])))
    cluster_info = cluster_info[cluster_info[:,1].argsort()]
    # print(f'clustering, min size: {cluster_info[0, 1]}, max_size: {cluster_info[-1, 1]}')

    clusters_labels = cluster_info[::-1][:args.num_clusters, 0]
    labels[np.in1d(labels, clusters_labels, invert=True)] = -1 # unclustered point
    # print('cluster info', cluster_info[::-1])
    return labels

def cluster_pcd(args, points, idxs_nonground):

    if args.if_hdbscan:
        raise ValueError("HDBSCAN not supported")
    else:
        labels_nonground = cluster_dbscan(args, points[idxs_nonground])

    # # # # visualize
    labels = np.zeros((len(points))) -1e8
    labels[idxs_nonground] = labels_nonground
    # visualize_pcd(points, labels, num_colors=100, title=f'cluster')
    # visualize_pcd_plotly(points, labels, num_colors=100)

    return labels
