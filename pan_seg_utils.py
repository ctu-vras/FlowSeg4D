import torch

def transform_pointcloud(points, transform_matrix):
    if not isinstance(transform_matrix, torch.Tensor):
        transform_matrix = torch.tensor(transform_matrix, dtype=points.dtype,
                                        device=points.device)
    points_tr = torch.cat((points, torch.ones((points.shape[0], 1))), axis=1)
    points_tr = torch.mm(transform_matrix, points_tr.T).T

    return points_tr[:, :3]