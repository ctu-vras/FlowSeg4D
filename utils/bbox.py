from typing import Tuple, Union

import torch
import numpy as np


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


def get_box_transform(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    transform = torch.eye(4)
    transform[:3, 3] = box2[:3] - box1[:3]
    transform[:3, :3] = torch.tensor(
        [
            [torch.cos(box2[6] - box1[6]), -torch.sin(box2[6] - box1[6]), 0],
            [torch.sin(box2[6] - box1[6]), torch.cos(box2[6] - box1[6]), 0],
            [0, 0, 1],
        ]
    )

    return transform


def transform_box(box: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    box_t = torch.zeros_like(box)
    box_t[:3] = torch.matmul(transform[:3, :3], box[:3]) + transform[:3, 3]
    box_t[3:5] = box[3:5]
    box_t[6] = box[6] + torch.atan2(transform[1, 0], transform[0, 0])

    return box_t
