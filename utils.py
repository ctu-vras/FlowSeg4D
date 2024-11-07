# -*- coding: utf-8 -*-

from typing import Union
from pathlib import Path

import torch
import numpy as np
import open3d as o3d


def load_pcd(file_path: Union[Path, str], dataset: str) -> torch.Tensor:
    if dataset == "waymo":
        pass  # TODO: Implement
    elif dataset == "av2":
        pass  # TODO: Implement
    else:
        raise NotImplementedError(f"Dataset {dataset} not supported")


def get_clusters(
    pcd: Union[torch.Tensor, np.ndarray], eps: float, min_points: int
) -> np.ndarray:
    if isinstance(pcd, torch.Tensor):
        pcd = pcd.cpu().numpy()
    assert pcd.ndim == 2, "Input tensor must have shape (N, 3)"
    assert pcd.shape[1] == 3, "Input tensor must have shape (N, 3)"

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd)
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))

    return labels
