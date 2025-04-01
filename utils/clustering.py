import torch
import hdbscan
import numpy as np
from sklearn.cluster import DBSCAN


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

        if config["clustering_method"] == "alpine":
            class_labels = clustering_alpine(points_np[mask], config)
        elif config["clustering_method"] == "hdbscan":
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
    cluster_info = np.array(list(zip(lbls[1:], counts[1:]))).reshape(-1, 2)
    cluster_info = cluster_info[cluster_info[:, 1].argsort()]

    clusters_labels = cluster_info[::-1][: config["num_clusters"], 0]
    labels[np.in1d(labels, clusters_labels, invert=True)] = -1

    return torch.tensor(labels, dtype=torch.int32, device=points.device)


def clustering_alpine(points, config):
    raise NotImplementedError


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
