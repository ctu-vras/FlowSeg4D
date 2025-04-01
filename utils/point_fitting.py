import sklearn

from typing import Union, Optional

import torch
import numpy as np
import open3d as o3d
from joblib import Parallel, delayed

from utils import rot_matrix_from_Euler


# TODO: Set hyperparameters, later change for json config file
TRANSLATION = 1  # Init translation range for ICP
N = 6  # Number of translation configurations
M = 10  # Number of rotation configurations
FITNESS_THRESHOLD = 0.3  # Fitness threshold for good match
DISTANCE_THRESHOLD = 0.4  # Symetric distance threshold for good match


def icp_match(
    source_pts: np.ndarray,
    target_pts: np.ndarray,
    t_x: float,
    t_y: float,
    rot_matrix: np.ndarray,
    threshold: float = 0.1,
    voxel_size: Optional[float] = None,
):
    """
    Run the ICP transformation between two point clouds
    Parameters
    ----------
    source_pts: np.ndarray
        the source point cloud
    target_pts: np.ndarray
        point cloud to be transformed
    t_x: float
        translation in x axis
    t_y: float
        translation in y axis
    rot_matrix: np.ndarray
        rotation matrix
    threshold: float
        threshold for convergence
    voxel_size: float
        voxel size for downsampling
    """
    # Set initial transformation
    init_transform = np.eye(4)
    init_transform[:3, :3] = rot_matrix
    init_transform[:3, 3] = np.array([t_x, t_y, 0])

    # Initialize the point clouds
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source_pts)
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target_pts)

    # If required, downsample the point clouds
    if voxel_size is not None:
        source_pcd.voxel_down_sample(voxel_size)
        target_pcd.voxel_down_sample(voxel_size)
        threshold = voxel_size * 2

    # Run the ICP transformation
    icp_result = o3d.pipelines.registration.registration_icp(
        source=source_pcd,
        target=target_pcd,
        max_correspondence_distance=threshold,
        init=init_transform,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )

    # Return the result in pickable format
    return {
        "fitness": icp_result.fitness,
        "inlier_rmse": icp_result.inlier_rmse,
        "transformation": icp_result.transformation,
    }


def icp_transform(
    source: Union[torch.Tensor, np.ndarray],
    sample: Union[torch.Tensor, np.ndarray],
    treshold: float = 0.1,
    voxel_size: Optional[float] = None,
):
    """
    ICP transformation between two point clouds
    Parameters
    ----------
    source: torch.Tensor | np.ndarray
        the source point cloud
    sample: torch.Tensor | np.ndarray
        point cloud to be transformed
    treshold: float
        treshold for convergence, automatically set to 2 * voxel_size if voxel_size is not None
    voxel_size: float
        voxel size for downsampling
    """
    if isinstance(source, torch.Tensor):
        source = source.cpu().numpy()
    if isinstance(sample, torch.Tensor):
        sample = sample.cpu().numpy()
    assert source.ndim == 2, "Source data must have shape (N, 3)"
    assert source.shape[1] == 3, "Source data must have shape (N, 3)"
    assert sample.ndim == 2, "Target data must have shape (M, 3)"
    assert sample.shape[1] == 3, "Target data must have shape (M, 3)"

    # Generate the translation and rotation matrices
    translate = np.linspace(-TRANSLATION, TRANSLATION, N)
    T_x, T_y = np.meshgrid(translate, translate)
    T_x, T_y = T_x.ravel(), T_y.ravel()

    rot_mat = [rot_matrix_from_Euler(0, 0, i * 2 * np.pi / M) for i in range(M)]

    # Initialize the point clouds and move the point clouds to their centroid
    source_cent = (np.min(source, axis=0) + np.max(source, axis=0)) / 2
    source_pcd = source - source_cent
    sample_cent = (np.min(sample, axis=0) + np.max(sample, axis=0)) / 2
    target_pcd = sample - sample_cent

    # Run the ICP transformation
    results = Parallel(n_jobs=-1)(
        delayed(icp_match)(
            source_pcd, target_pcd, T_x[i], T_y[i], rot_mat[j], treshold, voxel_size
        )
        for i in range(len(T_x))
        for j in range(len(rot_mat))
    )

    # Return the best result
    return max(results, key=lambda x: x["fitness"])


def icp_match_torch(source, target, n_iter=30):
    source_cent = (
        torch.min(source, dim=0).values + torch.max(source, dim=0).values
    ) / 2
    coords = source - source_cent
    target_cent = (
        torch.min(target, dim=0).values + torch.max(target, dim=0).values
    ) / 2
    coords_ref = target - target_cent

    if coords.shape[1] == 3:
        coords = torch.cat((coords, torch.ones(coords.shape[0], 1)), dim=1)
        coords_ref = torch.cat((coords_ref, torch.ones(coords_ref.shape[0], 1)), dim=1)

    for _ in range(n_iter):
        cdist = torch.cdist(coords, coords_ref)
        mindists, argmins = torch.min(cdist, dim=1)
        X = torch.linalg.lstsq(coords, coords_ref[argmins]).solution
        coords = coords @ X.T
        rmse = torch.sqrt(torch.mean(mindists**2))
        print(f"RMSE: {rmse:.4f}")
    return coords[:, :3] - source_cent


def good_match(
    source: Union[torch.Tensor, np.ndarray],
    sample: Union[torch.Tensor, np.ndarray],
    icp_result: dict,
    verbose: bool = False,
    save_name: Optional[str] = None,
    visualize: bool = False,
) -> bool:
    """
    Determine if the ICP match is good - objects are of the same class
    Parameters
    ----------
    source: torch.Tensor | np.ndarray
        the source point cloud
    sample: torch.Tensor | np.ndarray
        point cloud to be transformed
    icp_result: o3d.pipelines.registration.RegistrationResult
        the result of the ICP transformation
    verbose: bool
        print the fitness, RMSE, and symetric distance
    save_name: str
        the name of the file to save the point clouds
    visualize: bool
        visualize the point clouds
    """
    if isinstance(source, torch.Tensor):
        source = source.cpu().numpy()
    if isinstance(sample, torch.Tensor):
        sample = sample.cpu().numpy()
    assert source.ndim == 2, "Source data must have shape (N, 3)"
    assert source.shape[1] == 3, "Source data must have shape (N, 3)"
    assert sample.ndim == 2, "Target data must have shape (M, 3)"
    assert sample.shape[1] == 3, "Target data must have shape (M, 3)"

    # Initialize the point clouds and move the point clouds to their centroid
    source_pcd = o3d.geometry.PointCloud()
    source_cent = (np.min(source, axis=0) + np.max(source, axis=0)) / 2
    source_pcd.points = o3d.utility.Vector3dVector(source - source_cent)
    target_pcd = o3d.geometry.PointCloud()
    sample_cent = (np.min(sample, axis=0) + np.max(sample, axis=0)) / 2
    target_pcd.points = o3d.utility.Vector3dVector(sample - sample_cent)
    source_pcd.transform(icp_result["transformation"])

    # Compute the fitness, RMSE, and symetric distance
    fitness = icp_result["fitness"]
    rmse = icp_result["inlier_rmse"]
    source_to_target = np.mean(
        np.asarray(source_pcd.compute_point_cloud_distance(target_pcd))
    )
    target_to_source = np.mean(
        np.asarray(target_pcd.compute_point_cloud_distance(source_pcd))
    )
    symetric_distance = (source_to_target + target_to_source) / 2

    if verbose:
        print(f"Fitness: {fitness}")
        print(f"RMSE: {rmse}")
        print(f"Symetric distance: {symetric_distance}")

    # Color and save the point clouds
    if save_name is not None or visualize:
        source_pcd.paint_uniform_color([1, 0, 0])
        target_pcd.paint_uniform_color([0, 1, 0])
        merged_pcd = source_pcd + target_pcd
        if visualize:
            o3d.visualization.draw_geometries([merged_pcd])
        if save_name is not None:
            o3d.io.write_point_cloud(save_name, merged_pcd)

    # Determine if the match is good
    return fitness > FITNESS_THRESHOLD and symetric_distance < DISTANCE_THRESHOLD
