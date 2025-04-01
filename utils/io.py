import sklearn

from typing import Union
from pathlib import Path

import numpy as np
import pandas as pd
import open3d as o3d
from nuscenes import NuScenes
import pyarrow.feather as feather


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
