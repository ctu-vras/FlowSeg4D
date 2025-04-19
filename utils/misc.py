import copy
from dataclasses import dataclass
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
    feat: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes cluster centers (median) for a given class. If feat is provided, it computes
    the median for the feature tensor instead of the original points.

    Args:
        points (torch.Tensor): Input tensor of shape (N, D), where last two columns
                               represent class ID and cluster ID.
        class_id (int): The class ID to filter clusters for.
        feat (Optional[torch.Tensor]): Optional feature tensor of shape (N, M) to compute
                                       centers for. It could be flow - shape (N, 3), or
                                       some per point features - shape (N, M).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - Tensor of shape (num_clusters, 3 or M) containing computed median.
            - Tensor of unique cluster IDs.
    """
    class_mask = points[:, -2] == class_id
    clusters = torch.unique(points[class_mask, -1]).long()
    clusters = clusters[clusters != -1].sort()[0]

    if clusters.numel() == 0:
        return torch.empty(0, 3), clusters

    if feat is None:
        centers = torch.stack(
            [
                points[(class_mask) & (points[:, -1] == cluster_id), :3].median(dim=0).values
                for cluster_id in clusters
            ]
        )
    else:
        centers = torch.stack(
            [
                feat[(class_mask) & (points[:, -1] == cluster_id)].median(dim=0).values
                for cluster_id in clusters
            ]
        )

    return centers.double(), clusters

@dataclass
class Obj_cache:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.max_id = 0
        self.prev_instances = [dict() for _ in range(self.num_classes)]

    def add_instance(self, class_id, instance):
        self.prev_instances[class_id][instance.id] = copy.deepcopy(instance)

    def del_instance(self, class_id, instance_id):
        if instance_id in self.prev_instances[class_id].keys():
            del self.prev_instances[class_id][instance_id]
        else:
            print(f"Instance {instance_id} not found in class {class_id}.")

    def update_step(self):
        del_list = []
        for i in range(len(self.prev_instances)):
            for j in self.prev_instances[i].keys():
                self.prev_instances[i][j].life -= 1
                if self.prev_instances[i][j].life < 0:
                    del_list.append((i, j))
        for i, j in del_list:
            self.del_instance(i, j)

@dataclass
class Instance_data:
    id: int
    life: int
    center: torch.Tensor
    # bbox : torch.Tensor
    feature: torch.Tensor

    def __repr__(self):
        return f"Instance_data(id={self.id}, center={self.center}, life={self.life})"
