from typing import Union, Tuple, Optional

import yaml
import torch
import numpy as np


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


def get_centers_for_class(
    points: torch.Tensor,
    class_id: int,
    flow: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes cluster centers for a given class. If flow is provided, it computes
    the centers for the flow instead of the original points.

    Args:
        points (torch.Tensor): Input tensor of shape (N, D), where last two columns
                               represent class ID and cluster ID.
        class_id (int): The class ID to filter clusters for.
        flow (Optional[torch.Tensor]): Optional flow tensor of shape (N, D) to compute
                                       centers for.

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

    if flow is None:
        centers = torch.stack(
            [
                points[(class_mask) & (points[:, -1] == cluster_id), :3].mean(dim=0)
                for cluster_id in clusters
            ]
        )
    else:
        centers = torch.stack(
            [
                flow[(class_mask) & (points[:, -1] == cluster_id), :3].mean(dim=0)
                for cluster_id in clusters
            ]
        )

    return centers, clusters
