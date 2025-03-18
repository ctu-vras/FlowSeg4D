# -*- coding: utf-8 -*-

import sklearn

from pathlib import Path
from typing import Union, Optional, Tuple

import torch
import numpy as np
import pandas as pd
import open3d as o3d
from nuscenes import NuScenes
import pyarrow.feather as feather
from matplotlib import pyplot as plt


def load_pcd(file_path: Union[Path, str], dataset: str) -> np.ndarray:
    if dataset == "av2":
        pcd_data = feather.read_feather(file_path)
        pcd = np.c_[pcd_data["x"], pcd_data["y"], pcd_data["z"]]
    elif dataset == "waymo":
        raise NotImplementedError("Waymo dataset not yet supported")  # TODO: Implement
    elif dataset == "scala3":
        pcd = np.load(file_path, allow_pickle=True)["arr_0"].item()["pc"][:, :3]
    elif dataset == "pone":
        pcd = np.load(file_path, allow_pickle=True)["scan_list"]
    else:
        raise NotImplementedError(f"Dataset {dataset} not supported")

    return pcd


def load_pcd_file(file_path: Union[Path, str]) -> o3d.geometry.PointCloud:
    file = o3d.io.read_point_cloud(file_path)
    return file


def load_boxes(file_path: Union[Path, str], dataset: str) -> pd.DataFrame:
    if dataset == "av2":
        boxes = feather.read_feather(file_path)
    elif dataset == "waymo":
        raise NotImplementedError("Waymo dataset not yet supported")  # TODO: Implement
    elif dataset == "scala3":
        raise ValueError("Scala3 dataset not yet supported")
    elif dataset == "pone":
        raise ValueError("PONE dataset not yet supported")
    else:
        raise NotImplementedError(f"Dataset {dataset} not supported")
    return boxes


def load_nuScenes(
    data_root: Union[Path, str], split: str = "mini", verbose: bool = False
) -> NuScenes:
    nusc = NuScenes(version=f"v1.0-{split}", dataroot=str(data_root), verbose=verbose)
    return nusc


def quaternion_to_yaw(q: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    device = q.device if isinstance(q, torch.Tensor) else "cpu"
    if isinstance(q, np.ndarray):
        q = torch.tensor(q, device=device)
    assert q.shape[-1] == 4, "Input data must have shape (..., 4)"

    q = q / torch.norm(q, dim=-1, keepdim=True)
    yaw = torch.atan2(
        2 * (q[..., 0] * q[..., 3] + q[..., 1] * q[..., 2]),
        1 - 2 * (q[..., 2] ** 2 + q[..., 3] ** 2),
    )
    return yaw


def get_clusters(
    pcd_in: Union[torch.Tensor, np.ndarray], eps: float, min_points: int
) -> np.ndarray:
    if isinstance(pcd_in, torch.Tensor):
        pcd_in = pcd_in.cpu().numpy()
    assert pcd_in.ndim == 2, "Input data must have shape (N, 3)"
    assert pcd_in.shape[1] == 3, "Input data must have shape (N, 3)"

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_in)
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))

    return labels


def visualize_pcd(
    pcd_in: Union[torch.Tensor, np.ndarray, o3d.geometry.PointCloud],
    labels: Optional[np.ndarray] = None,
) -> None:
    if not isinstance(pcd_in, o3d.geometry.PointCloud):
        if isinstance(pcd_in, torch.Tensor):
            pcd_in = pcd_in.cpu().numpy()
        assert pcd_in.ndim == 2, "Input data must have shape (N, 3)"
        assert pcd_in.shape[1] == 3, "Input data must have shape (N, 3)"
        assert pcd_in.shape[0] > 0, "Input data must have at least one point"
        if labels is not None:
            assert (
                pcd_in.shape[0] == labels.shape[0]
            ), "Number of points must match number of labels"

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_in)
    else:
        pcd = pcd_in
    if labels is not None:
        colors = plt.get_cmap("hsv")(labels / (labels.max() if labels.max() > 0 else 1))
        colors[labels < 0] = 0  # Set noise points to black
        pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([pcd])


def visualize_flow(points, labels=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    if labels is None:
        pass
    else:
        colors = labels / np.max(labels)
        colors[labels < 0] = 0
        pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])


def get_points_in_box(
    pcd: Union[torch.Tensor, np.ndarray], box: Union[torch.Tensor, np.ndarray]
):
    device = pcd.device if isinstance(pcd, torch.Tensor) else "cpu"
    if isinstance(pcd, np.ndarray):
        pcd = torch.tensor(pcd, device=device)
    if isinstance(box, np.ndarray):
        box = torch.tensor(box, device=device)
    assert pcd.ndim == 2, "Input data must have shape (N, 3)"
    assert pcd.shape[1] == 3, "Input data must have shape (N, 3)"
    assert box.shape == (7,), "Box must have shape (7,)"

    mask = get_box_mask(pcd, box)

    return pcd[mask]


def rot_matrix_from_Euler(alpha: float, beta: float, gamma: float) -> np.ndarray:
    """
    Get the rotation matrix from Euler angles
    Parameters
    ----------
    alpha: float
        rotation angle around x-axis
    beta: float
        rotation angle around y-axis
    gamma: float
        rotation angle around z-axis
    Returns
    -------
    np.ndarray
        rotation matrix
    """
    sa, ca = np.sin(alpha), np.cos(alpha)
    sb, cb = np.sin(beta), np.cos(beta)
    sg, cg = np.sin(gamma), np.cos(gamma)
    ret = np.array(
        [
            [cb * cg, sa * sb * cg - ca * sg, sa * sg + ca * sb * cg],
            [cb * sg, ca * cg + sa * sb * sg, ca * sb * sg - sa * cg],
            [-sb, sa * cb, ca * cb],
        ]
    )
    return ret


def get_box_mask(
    pcd: Union[torch.Tensor, np.ndarray], box: Union[torch.Tensor, np.ndarray]
) -> torch.Tensor:
    device = pcd.device if isinstance(pcd, torch.Tensor) else "cpu"
    if isinstance(pcd, np.ndarray):
        pcd = torch.tensor(pcd, device=device)
    if isinstance(box, np.ndarray):
        box = torch.tensor(box, device=device)
    assert pcd.ndim == 2, "Input data must have shape (N, 3)"
    assert pcd.shape[1] == 3, "Input data must have shape (N, 3)"
    assert box.shape == (7,), "Box must have shape (7,)"

    # Get rotation matrix and translation vector
    rot_mat = torch.tensor(
        [
            [torch.cos(box[6]), -torch.sin(box[6])],
            [torch.sin(box[6]), torch.cos(box[6])],
        ],
        device=device,
    )
    translation = torch.tensor([box[0], box[1]], device=device)

    # Transform pcd to box frame
    pcd_r = (pcd[:, :2] - translation) @ rot_mat

    # Get mask of points inside the box
    mask = (
        (-box[3] / 2 <= pcd_r[:, 0])
        & (pcd_r[:, 0] <= box[3] / 2)
        & (-box[4] / 2 <= pcd_r[:, 1])
        & (pcd_r[:, 1] <= box[4] / 2)
    )

    return mask


def remove_ego_vehicle(
    pcd: Union[torch.Tensor, np.ndarray], dataset: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = pcd.device if isinstance(pcd, torch.Tensor) else "cpu"
    if isinstance(pcd, np.ndarray):
        pcd = torch.tensor(pcd, device=device)
    assert pcd.ndim == 2, "Input data must have shape (N, M)"

    if dataset == "av2":
        ego_box = torch.tensor([0, 0, 0, 4.841, 1.834, 1.445, 0])
    elif dataset == "nuscenes":
        ego_box = torch.tensor([0, 0, 0, 4.084, 1.730, 1.562, 0])
    else:
        raise ValueError(f"Dataset {dataset} not supported")

    mask = get_box_mask(pcd[:, :3], ego_box)

    return pcd[~mask], ~mask
