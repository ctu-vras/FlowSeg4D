import sklearn

from typing import Optional, Union

import torch
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


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


def vis_box_bev(box: torch.Tensor, color: str) -> None:
    corners = np.array(
        [
            [-box[3] / 2, -box[4] / 2],
            [-box[3] / 2, box[4] / 2],
            [box[3] / 2, box[4] / 2],
            [box[3] / 2, -box[4] / 2],
        ]
    )
    rot_mat = np.array(
        [
            [np.cos(box[6]), -np.sin(box[6])],
            [np.sin(box[6]), np.cos(box[6])],
        ]
    )
    corners @= rot_mat.T
    corners += np.array([box[0], box[1]])
    corners = np.vstack([corners, corners[0]])
    plt.plot(corners[:, 0], corners[:, 1], color=color)
