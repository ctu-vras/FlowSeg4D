import os
import time
from typing import Optional, Union

import torch
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from utils.misc import load_config

MAX_INST = 75


def visualize_scene(config: dict, pcd_dir: str, labels_dir: str) -> None:
    pcd_files = sorted(os.listdir(pcd_dir))
    lab_files = sorted(os.listdir(labels_dir))
    assert len(pcd_files) == len(lab_files), (
        f"Mismatch between point cloud (num: {len(pcd_files)}) and label files (num: {len(lab_files)})."
    )

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    pcd = o3d.geometry.PointCloud()
    geometry_added = False

    if config["dataset"] == "semantic_kitti":
        sem_kitti_conf = load_config("configs/semantic-kitti.yaml")
        mapper = np.vectorize(sem_kitti_conf["learning_map"].__getitem__)

    for pcd_file, lab_file in zip(pcd_files, lab_files):
        start_time = time.time_ns()
        if config["dataset"] == "pone":
            points = np.load(os.path.join(pcd_dir, pcd_file))["pcd"][:, :3]
        elif config["dataset"] == "semantic_kitti":
            points = np.fromfile(
                os.path.join(pcd_dir, pcd_file), dtype=np.float32
            ).reshape(-1, 4)[:, :3]
        else:
            raise ValueError(f"Dataset {config['dataset']} not supported.")
        pcd.points = o3d.utility.Vector3dVector(points)

        # Load labels
        if config["instances"]:
            labels = (
                np.fromfile(os.path.join(labels_dir, lab_file), dtype=np.uint32)
                & 0xFFFF0000
            )
            labels = (labels >> 16).astype(np.int16)
        else:
            labels = (
                np.fromfile(os.path.join(labels_dir, lab_file), dtype=np.uint32)
                & 0xFFFF
            ).astype(np.int16)
            if config["dataset"] == "semantic_kitti":
                labels = mapper(labels) - 1

        # Assign colors based on labels
        if config["colors"] is None or config["instances"]:
            colors = plt.get_cmap("hsv")((labels % MAX_INST) / MAX_INST)
            colors[labels == 0] = 0
        else:
            colors = config["colors"][labels]
            colors[labels == -1] = 0
        pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

        # Visualize the point cloud
        if not geometry_added:
            vis.add_geometry(pcd)
            geometry_added = True
            time.sleep(2)
        else:
            vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        duration = time.time_ns() - start_time
        if (duration // 1000000) < config["fps"]:
            time.sleep(config["fps"] - (duration // 1000000))


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
            assert pcd_in.shape[0] == labels.shape[0], (
                "Number of points must match number of labels"
            )

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
