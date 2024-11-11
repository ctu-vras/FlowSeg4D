# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Union, Optional

import torch
import numpy as np
import pandas as pd
import open3d as o3d
import pyarrow.feather as feather
from matplotlib import pyplot as plt


def load_pcd(file_path: Union[Path, str], dataset: str) -> np.ndarray:
    if dataset == "av2":
        pcd = feather.read_feather(file_path)
    elif dataset == "waymo":
        raise NotImplementedError("Waymo dataset not yet supported")  # TODO: Implement
    elif dataset == "scala3":
        pcd = np.load(file_path, allow_pickle=True)["arr_0"].item()["pc"][:, :3]
    elif dataset == "pone":
        pcd = np.load(file_path, allow_pickle=True)["scan_list"]
    else:
        raise NotImplementedError(f"Dataset {dataset} not supported")

    return pcd


def load_boxes(file_path: Union[Path, str], dataset: str) -> pd.DataFrame:
    if dataset == "av2":
        boxes = feather.read_feather(file_path)
    elif dataset == "waymo":
        raise NotImplementedError("Waymo dataset not yet supported")  # TODO: Implement
    elif dataset == "scala3":
        raise ValueError("Scala3 dataset does not have boxes")
    elif dataset == "pone":
        raise ValueError("Scala3 dataset does not have boxes")
    else:
        raise NotImplementedError(f"Dataset {dataset} not supported")
    return boxes


def quaternion_to_yaw(q: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    if isinstance(q, torch.Tensor):
        q = q.cpu().numpy()
    assert q.shape[-1] == 4, "Input tensor must have shape (..., 4)"
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    yaw = np.arctan2(
        2 * (q[..., 3] * q[..., 2] + q[..., 0] * q[..., 1]),
        1 - 2 * (q[..., 1] ** 2 + q[..., 2] ** 2),
    )
    return yaw


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
