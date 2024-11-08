#!/bin/bash/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import Union, Tuple

import tqdm
import torch
import numpy as np
import open3d as o3d
import pypatchworkpp


def parse_args():
    parser = argparse.ArgumentParser(description="Ground Removal")

    parser.add_argument(
        "--data_root", type=str, required=True, help="Path to the root data directory"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        help="Path to save the results, if not specified, the results will be saved in <data_dir>_xyz",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="av2",
        help="Name of the dataset, default: av2",
    )
    parser.add_argument("--verbose", action="store_true", help="Print verbose messages")
    parser.add_argument(
        "--visualize", action="store_true", help="Visualize the results"
    )

    return parser.parse_args()


class GroundRemoval:
    def __init__(self, args: argparse.Namespace, standalone: bool = False):
        params = pypatchworkpp.Parameters()
        params.verbose = args.verbose
        params.enable_RNR = False

        self.patchwork = pypatchworkpp.patchworkpp(params)

        self.standalone = standalone
        if standalone:
            self.save_dir = Path(args.save_dir) if args.save_dir is not None else None
            self.data_dir = None
        else:
            self.save_dir = None
            self.data_dir = Path(args.data_root) / args.dataset
        self.args = args

    def _read_pcd(self, pcd_path: Union[str, Path]) -> np.ndarray:
        """
        Read a point cloud file and return the point cloud data,
        it is prepared for the data format of scala3 dataset provided by Valeo

        :param Union[str,Path] pcd_path:
            Path to the point cloud file

        :return: np.ndarray:
            Point cloud data
        """
        if self.args.dataset == "scala3":
            pcd = np.load(pcd_path, allow_pickle=True)["arr_0"].item()["pc"][:, :3]
        elif self.args.dataset == "pone":
            pcd = np.load(pcd_path, allow_pickle=True)["scan_list"]
        else:
            raise ValueError(f"Unsupported dataset: {self.args.dataset}")

        return pcd

    def _save_xyz(self, file_name: Union[str, Path]) -> None:
        """
        Save the nonground points to a file in XYZ format

        :param Union[str,Path] file_name:
            Name of the file to save the nonground points
        """
        save_dir = (
            self.save_dir
            if self.save_dir is not None
            else self.data_dir.parent / (self.args.dataset + "_xyz")
        )
        save_dir.mkdir(parents=True, exist_ok=True)
        save_data = np.concatenate(
            [
                self.patchwork.getNonground(),
                self.patchwork.getNongroundIndices().reshape(-1, 1),
            ],
            axis=1,
        )
        np.savetxt(
            save_dir / f"{file_name}_nonground.xyz",
            save_data,
            fmt="%.6f",
        )

    def _prep_pone(self, data: np.ndarray) -> np.ndarray:
        """
        Prepare the point cloud data of the PONE dataset

        :param np.ndarray data:
            Point cloud data

        :return: np.ndarray:
            Prepared point cloud data
        """
        pcd = np.concatenate([data["x"], data["z"].reshape(-1, 1)], axis=1)
        return pcd

    def run(self) -> None:
        """
        Run the ground removal algorithm, save the results as .xyz files
        """
        if self.standalone:
            raise ValueError("This method is not supported in standalone mode")

        if self.args.dataset == "av2":
            raise NotImplementedError("Dataset 'av2' not yet supported")
        elif self.args.dataset == "scala3":
            files = sorted(self.data_dir.glob("*.npz"))
            pcd_count = len(files)
        elif self.args.dataset == "pone":
            files = sorted(self.data_dir.glob("*PCD.npz"))[0]
            data = self._read_pcd(files)
            pcd_count = data.shape[0]
        else:
            raise ValueError(f"Unsupported dataset: {self.args.dataset}")

        for idx in tqdm.tqdm(range(pcd_count)):
            pcd = (
                self._read_pcd(files[idx])
                if self.args.dataset == "scala3"
                else self._prep_pone(data[idx])
            )
            self.patchwork.estimateGround(pcd)

            if self.args.visualize:
                self._visualize()

            save_name = (
                files[idx].stem
                if self.args.dataset == "scala3"
                else f"{files.stem}_{idx:04d}"
            )
            self._save_xyz(save_name)

    def run_individual_file(
        self, file_path: Union[str, Path]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run the ground removal algorithm on a single point cloud file

        :param Union[str, Path] file_path:
            Path to the point cloud file

        :return: Tuple[np.ndarray, np.ndarray]:
            Nonground points, Nonground indices
        """
        if self.args.dataset != "scala3":
            raise ValueError(f"Unsupported dataset: {self.args.dataset}")

        full_pcd = self._read_pcd(file_path)
        pcd = full_pcd[:, :3]

        self.patchwork.estimateGround(pcd)

        return self.patchwork.getNonground(), self.patchwork.getNongroundIndices()

    def run_individual_scan(
        self, pcd: Union[torch.Tensor, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run the ground removal algorithm on a given point cloud data

        :param Union[torch.Tensor, np.ndarray] pcd:
            Point cloud data

        :return: Tuple[np.ndarray, np.ndarray]:
            Nonground points, Nonground indices
        """
        if isinstance(pcd, torch.Tensor):
            pcd = pcd.cpu().numpy()

        self.patchwork.estimateGround(pcd)

        return self.patchwork.getNonground(), self.patchwork.getNongroundIndices()

    def _visualize(self) -> None:
        """
        Visualize the ground and nonground points along with the surface normals
        """
        ground = self.patchwork.getGround()
        nonground = self.patchwork.getNonground()
        time_taken = self.patchwork.getTimeTaken()

        # Get centers and normals for patches
        centers = self.patchwork.getCenters()
        normals = self.patchwork.getNormals()

        print("Original Points  #: ", (ground.shape[0] + nonground.shape[0]))
        print("Ground Points    #: ", ground.shape[0])
        print("Nonground Points #: ", nonground.shape[0])
        print("Time Taken : ", time_taken / 1000000, "(sec)")
        print("Press ... \n")
        print("\t H  : help")
        print("\t N  : visualize the surface normals")
        print("\tESC : close the Open3D window")

        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(width=600, height=400)

        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()

        ground_o3d = o3d.geometry.PointCloud()
        ground_o3d.points = o3d.utility.Vector3dVector(ground)
        ground_o3d.colors = o3d.utility.Vector3dVector(
            np.array(
                [[1.0, 0.0, 0.0] for _ in range(ground.shape[0])], dtype=float
            )  # RGB
        )

        nonground_o3d = o3d.geometry.PointCloud()
        nonground_o3d.points = o3d.utility.Vector3dVector(nonground)
        nonground_o3d.colors = o3d.utility.Vector3dVector(
            np.array(
                [[0.0, 1.0, 0.0] for _ in range(nonground.shape[0])], dtype=float
            )  # RGB
        )

        centers_o3d = o3d.geometry.PointCloud()
        centers_o3d.points = o3d.utility.Vector3dVector(centers)
        centers_o3d.normals = o3d.utility.Vector3dVector(normals)
        centers_o3d.colors = o3d.utility.Vector3dVector(
            np.array(
                [[0.0, 0.0, 1.0] for _ in range(centers.shape[0])], dtype=float
            )  # RGB
        )

        vis.add_geometry(mesh)
        vis.add_geometry(ground_o3d)
        vis.add_geometry(nonground_o3d)
        vis.add_geometry(centers_o3d)

        vis.run()
        vis.destroy_window()


if __name__ == "__main__":
    args = parse_args()
    ground_removal = GroundRemoval(args)

    nonground = ground_removal.run()
