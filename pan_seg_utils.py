from typing import Union

import torch
import hdbscan
import numpy as np
from sklearn.cluster import DBSCAN

def transform_pointcloud(points: torch.Tensor, transform_matrix: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    '''
    Returns the transformed pointcloud given the transformation matrix.

    Args:
        points (torch.Tensor): The input pointcloud of shape (N, 3)
        transform_matrix (torch.Tensor or np.ndarray): The transformation matrix of shape (4, 4)

    Returns:
        torch.Tensor: The transformed pointcloud of shape (N, 3)
    '''
    assert points.dim() == 2, 'The input pointcloud should have shape (N, M)'
    assert points.shape[1] == 3, 'The input pointcloud should have shape (N, 3)'

    if not isinstance(transform_matrix, torch.Tensor):
        transform_matrix = torch.tensor(transform_matrix, dtype=points.dtype,
                                        device=points.device)

    points_tr = torch.cat((points, torch.ones((points.shape[0], 1))), axis=1)
    points_tr = torch.mm(transform_matrix, points_tr.T).T

    return points_tr[:, :3]

def get_semantic_clustering(points: torch.Tensor, config: dict) -> torch.Tensor:
    """
    Perform semantic clustering on input points using DBSCAN.

    Args:
        points (torch.Tensor): Tensor of shape (N, D) where last column contains class labels.
        config (dict): Dictionary with clustering parameters

    Returns:
        torch.Tensor: Cluster labels of shape (N,).
    """
    points_np = points.cpu().numpy()
    labels = np.full(points.shape[0], -1, dtype=np.int32)
    sem_classes = int(points[:, -1].max().item()) + 1
    cluster_id = 0

    for i in range(sem_classes):
        mask = points_np[:, -1] == i
        if mask.sum() < config['min_cluster_size']:
            continue
        
        if config['clustering_method'] == 'hdbscan':
            class_labels = clustering_hdbscan(points_np[mask], config)
        else:
            class_labels = clustering_dbscan(points_np[mask], config)

        unique_labels = np.unique(class_labels)
        labels[mask] = class_labels + cluster_id
        
        cluster_id += (len(unique_labels) - 1) if -1 in unique_labels else len(unique_labels)

    lbls, counts = np.unique(labels, return_counts=True)
    cluster_info = np.array(list(zip(lbls[1:], counts[1:])))
    cluster_info = cluster_info[cluster_info[:,1].argsort()]

    clusters_labels = cluster_info[::-1][:config["num_clusters"], 0]
    labels[np.in1d(labels, clusters_labels, invert=True)] = -1

    return torch.tensor(labels, dtype=torch.int32, device=points.device)

def clustering_hdbscan(points, config):
    clusterer = hdbscan.HDBSCAN(algorithm="best", approx_min_span_tree=True, gen_min_span_tree=True,
                                metric="euclidean", min_cluster_size=config["min_cluster_size"], min_samples=None)
    clusterer.fit(points[:, :3])

    return clusterer.labels_.copy()

def clustering_dbscan(points, config):
    db = DBSCAN(eps=config['epsilon'], min_samples=config['min_cluster_size'])
    db.fit(points[:, :3])

    return db.labels_.copy()
