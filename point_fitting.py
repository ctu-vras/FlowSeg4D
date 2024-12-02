import numpy as np
import open3d as o3d

from utils import rot_matrix_from_Euler


def icp_transform(source: np.ndarray, sample: np.ndarray, treshold: float = 0.1):
    """
    ICP transformation between two point clouds
    Parameters
    ----------
    source: np.ndarray
        the source point cloud
    sample: np.ndarray
        point cloud to be transformed
    treshold: float
        treshold for convergence
    """
    N = 20
    translate = np.linspace(-2, 2, N)

    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source - np.mean(source, axis=0))
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(sample - np.mean(sample, axis=0))
    init_transform = np.eye(4)

    # source_pcd.paint_uniform_color([1, 0, 0])
    # target_pcd.paint_uniform_color([0, 1, 0])
    # o3d.visualization.draw_geometries([source_pcd, target_pcd])

    best_icp_result = None
    for i in range(N):
        for j in range(N):
            for k in range(N):
                init_transform[:3, 3] = np.array([translate[i], 0, translate[j]])
                init_transform[:3, :3] = rot_matrix_from_Euler(0, 0, k * 2 * np.pi / N)
                icp_result = o3d.pipelines.registration.registration_icp(
                    source=source_pcd,
                    target=target_pcd,
                    max_correspondence_distance=treshold,
                    init=init_transform,
                    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                )
                if (
                    best_icp_result is None
                    or icp_result.fitness > best_icp_result.fitness
                ):
                    best_icp_result = icp_result

    return best_icp_result


def good_match(
    source: np.ndarray, sample: np.ndarray, icp_result, treshold: float = 0.1
) -> bool:
    """
    Get the good match between two point clouds
    Parameters
    ----------
    source: np.ndarray
        the source point cloud
    sample: np.ndarray
        point cloud to be transformed
    icp_result: o3d.pipelines.registration.RegistrationResult
        the result of the ICP transformation
    """
    fitness = icp_result.fitness
    rmse = icp_result.inlier_rmse

    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source - np.mean(source, axis=0))
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(sample - np.mean(sample, axis=0))
    source_pcd.transform(icp_result.transformation)
    print(icp_result.transformation)

    source_to_target = np.mean(
        np.asarray(source_pcd.compute_point_cloud_distance(target_pcd))
    )
    target_to_source = np.mean(
        np.asarray(target_pcd.compute_point_cloud_distance(source_pcd))
    )
    symetric_distance = (source_to_target + target_to_source) / 2

    print(f"Fitness: {fitness}")
    print(f"RMSE: {rmse}")
    print(f"Symetric distance: {symetric_distance}")

    # source_pcd.paint_uniform_color([1, 0, 0])
    # target_pcd.paint_uniform_color([0, 1, 0])
    # o3d.visualization.draw_geometries([source_pcd, target_pcd])

    return fitness > 0.3 and symetric_distance < 0.4
