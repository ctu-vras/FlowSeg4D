from typing import Union

import torch
import numpy as np


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
