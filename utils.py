# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Union, Optional

import torch
import numpy as np
import open3d as o3d
import pyarrow.feather as feather
from matplotlib import pyplot as plt


def load_pcd(file_path: Union[Path, str], dataset: str) -> torch.Tensor:
    if dataset == "waymo":
        raise NotImplementedError("Waymo dataset not yet supported")  # TODO: Implement
    elif dataset == "av2":
        pcd = feather.read_feather(file_path)
        pcd = np.c_[pcd["x"], pcd["y"], pcd["z"]]
    else:
        raise NotImplementedError(f"Dataset {dataset} not supported")

    return pcd


def get_clusters(
    pcd_in: Union[torch.Tensor, np.ndarray], eps: float, min_points: int
) -> np.ndarray:
    if isinstance(pcd_in, torch.Tensor):
        pcd_in = pcd_in.cpu().numpy()
    assert pcd_in.ndim == 2, "Input tensor must have shape (N, 3)"
    assert pcd_in.shape[1] == 3, "Input tensor must have shape (N, 3)"

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_in)
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))

    return labels


def visualize_pcd(
    pcd_in: Union[torch.Tensor, np.ndarray], labels: Optional[np.ndarray]
) -> None:
    if isinstance(pcd_in, torch.Tensor):
        pcd_in = pcd_in.cpu().numpy()
    assert pcd_in.ndim == 2, "Input tensor must have shape (N, 3)"
    assert pcd_in.shape[1] == 3, "Input tensor must have shape (N, 3)"
    if labels is not None:
        assert (
            pcd_in.shape[0] == labels.shape[0]
        ), "Number of points must match number of labels"
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_in)
    if labels is not None:
        colors = plt.get_cmap("tab20")(
            labels / (labels.max() if labels.max() > 0 else 1)
        )
        colors[labels < 0] = 0  # Set noise points to black
        pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([pcd])
