from typing import Union, Tuple, Optional

import yaml
import torch
import hdbscan
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.optimize import linear_sum_assignment


def load_model_config(file):
    with open(file, "r") as f:
        config = yaml.safe_load(f)
    return config


def transform_pointcloud(
    points: torch.Tensor, transform_matrix: Union[torch.Tensor, np.ndarray]
) -> torch.Tensor:
    """
    Returns the transformed pointcloud given the transformation matrix.

    Args:
        points (torch.Tensor): The input pointcloud of shape (N, 3)
        transform_matrix (torch.Tensor or np.ndarray): The transformation matrix of shape (4, 4)

    Returns:
        torch.Tensor: The transformed pointcloud of shape (N, 3)
    """
    assert points.dim() == 2, "The input pointcloud should have shape (N, M)"
    assert points.shape[1] == 3, "The input pointcloud should have shape (N, 3)"

    if not isinstance(transform_matrix, torch.Tensor):
        transform_matrix = torch.tensor(
            transform_matrix, dtype=points.dtype, device=points.device
        )

    points_tr = torch.cat(
        (points, torch.ones((points.shape[0], 1), device=points.device)), axis=1
    )
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
    cluster_id = 0

    for class_id in config["fore_classes"]:
        mask = points_np[:, -1] == class_id
        if mask.sum() < config["min_cluster_size"]:
            continue

        if config["clustering_method"] == "hdbscan":
            class_labels = clustering_hdbscan(points_np[mask], config)
        else:
            if config["clustering_method"] != "dbscan":
                print(
                    f"Invalid clustering method: {config['clustering_method']}. Using DBSCAN instead."
                )
            class_labels = clustering_dbscan(points_np[mask], config)

        # merge labels, outliers are labeled as -1
        updated_labels = class_labels + cluster_id
        updated_labels[class_labels == -1] = -1
        labels[mask] = updated_labels

        unique_labels = np.unique(class_labels)
        cluster_id += (
            (len(unique_labels) - 1) if -1 in unique_labels else len(unique_labels)
        )

    # keep only the top clusters
    lbls, counts = np.unique(labels, return_counts=True)
    cluster_info = np.array(list(zip(lbls[1:], counts[1:])))
    cluster_info = cluster_info[cluster_info[:, 1].argsort()]

    clusters_labels = cluster_info[::-1][: config["num_clusters"], 0]
    labels[np.in1d(labels, clusters_labels, invert=True)] = -1

    return torch.tensor(labels, dtype=torch.int32, device=points.device)


def clustering_hdbscan(points, config):
    clusterer = hdbscan.HDBSCAN(
        algorithm="best",
        approx_min_span_tree=True,
        gen_min_span_tree=True,
        metric="euclidean",
        min_cluster_size=config["min_cluster_size"],
        min_samples=None,
    )

    if isinstance(points, torch.Tensor):
        points_ = points.cpu().numpy().copy()
    else:
        points_ = points.copy()
    clusterer.fit(points_[:, :3])

    return clusterer.labels_.copy()


def clustering_dbscan(points, config):
    clusterer = DBSCAN(eps=config["epsilon"], min_samples=config["min_cluster_size"])

    if isinstance(points, torch.Tensor):
        points_ = points.cpu().numpy().copy()
    else:
        points_ = points.copy()
    clusterer.fit(points_[:, :3])

    return clusterer.labels_.copy()


def get_centers_for_class(
    points: torch.Tensor,
    class_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes cluster centers for a given class.

    Args:
        points (torch.Tensor): Input tensor of shape (N, D), where last two columns
                               represent class ID and cluster ID.
        class_id (int): The class ID to filter clusters for.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - Tensor of shape (num_clusters, 3) containing cluster centers.
            - Tensor of unique cluster IDs.
    """
    class_mask = points[:, -2] == class_id
    clusters = torch.unique(points[class_mask, -1]).long()
    clusters = clusters[clusters != -1].sort()[0]

    if clusters.numel() == 0:
        return torch.empty(0, 3), clusters

    centers = torch.stack(
        [
            points[(class_mask) & (points[:, -1] == cluster_id), :3].mean(dim=0)
            for cluster_id in clusters
        ]
    )

    return centers, clusters


def association(
    points_t1: torch.Tensor,
    points_t2: torch.Tensor,
    config: dict,
    prev_ind: Optional[torch.Tensor] = None,
    ind_cache: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    indices_t1 = torch.zeros(points_t1.shape[0], dtype=torch.int32)
    indices_t2 = torch.zeros(points_t2.shape[0], dtype=torch.int32)

    curr_id = 1 if ind_cache is None else ind_cache["max_id"] + 1

    for class_id in config["fore_classes"]:
        centers_t1, clusters_t1 = get_centers_for_class(points_t1, class_id)
        centers_t2, clusters_t2 = get_centers_for_class(points_t2, class_id)

        class_mask_t1 = points_t1[:, -2] == class_id
        class_mask_t2 = points_t2[:, -2] == class_id

        if clusters_t1.numel() == 0 and clusters_t2.numel() == 0:
            continue

        if clusters_t1.numel() == 0:
            for cluster_id in clusters_t2:
                mask = (class_mask_t2) & (points_t2[:, -1] == cluster_id)
                indices_t2[mask] = curr_id
                curr_id += 1
            continue

        if clusters_t2.numel() == 0:
            for cluster_id in clusters_t1:
                mask = (class_mask_t1) & (points_t1[:, -1] == cluster_id)
                if prev_ind is None:
                    indices_t1[mask] = curr_id
                    curr_id += 1
                else:
                    indices_t1[mask] = prev_ind[mask][0]
            continue

        dists = torch.cdist(centers_t1, centers_t2)
        # associate using hungarian matching
        # TODO: use different algorithm allowing for backpropagation
        row_ind, col_ind = linear_sum_assignment(dists.cpu().numpy())
        used_row, used_col = set(row_ind), set(col_ind)
        for i, j in zip(row_ind, col_ind):
            mask_t1 = (class_mask_t1) & (points_t1[:, -1] == clusters_t1[i])
            mask_t2 = (class_mask_t2) & (points_t2[:, -1] == clusters_t2[j])
            if dists[i, j] > config["max_dist"]:  # threshold for association
                indices_t1[mask_t1] = curr_id if prev_ind is None else prev_ind[mask_t1][0]
                curr_id += 1 if prev_ind is None else 0
                indices_t2[mask_t2] = curr_id
                curr_id += 1
            else:
                id_val = curr_id if prev_ind is None else prev_ind[mask_t1][0]
                indices_t1[mask_t1] = id_val
                indices_t2[mask_t2] = id_val
                curr_id += 1 if prev_ind is None else 0

        if centers_t1.shape[0] > centers_t2.shape[0]:
            for i, cluster_id in enumerate(clusters_t1):
                if i in used_row:
                    continue
                mask = (class_mask_t1) & (points_t1[:, -1] == cluster_id)
                if prev_ind is None:
                    indices_t1[mask] = curr_id
                    curr_id += 1
                else:
                    indices_t1[mask] = prev_ind[mask][0]
        elif centers_t1.shape[0] < centers_t2.shape[0]:
            for j, cluster_id in enumerate(clusters_t2):
                if j in used_col:
                    continue
                mask = (class_mask_t2) & (points_t2[:, -1] == cluster_id)
                indices_t2[mask] = curr_id
                curr_id += 1
        else:
            continue

    indices_t1 = indices_t1.to(points_t1.device)
    indices_t2 = indices_t2.to(points_t2.device)
    return indices_t1, indices_t2
