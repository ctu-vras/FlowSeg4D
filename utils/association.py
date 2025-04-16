from typing import Optional, Tuple

import torch
from scipy.optimize import linear_sum_assignment

from utils.misc import get_centers_for_class


def association(
    points_t1: torch.Tensor,
    points_t2: torch.Tensor,
    config: dict,
    prev_ind: Optional[torch.Tensor] = None,
    ind_cache: Optional[dict] = None,
    flow: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    This function performs association between two sets of points.
    It uses the Hungarian algorithm to find the optimal assignment of points
    from the first set to the second set based on distance of centers of clusters.

    Args:
        points_t1 (torch.Tensor): Data for time t, ego compensated xyz + semantic class + cluster id.
        points_t2 (torch.Tensor): Data for time t+1, ego compensated xyz + semantic class + cluster id.
        config (dict): Configuration dictionary containing parameters for association.
        prev_ind (Optional[torch.Tensor]): Previous indices for association.
        ind_cache (Optional[dict]): Cache for previous indices.
        flow (Optional[torch.Tensor]): Flow tensor for adjusting point positions.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The updated indices for points in both sets.
    """
    indices_t1 = torch.zeros(points_t1.shape[0], dtype=torch.int32)
    indices_t2 = torch.zeros(points_t2.shape[0], dtype=torch.int32)

    curr_id = 1 if ind_cache is None else ind_cache["max_id"] + 1

    for class_id in config["fore_classes"]:
        # Get the centers of clusters for the current class
        centers_t1, clusters_t1 = get_centers_for_class(points_t1, class_id)
        centers_t2, clusters_t2 = get_centers_for_class(points_t2, class_id)

        # If flow is provided, adjust the centers of t1
        if flow is not None:
            flow_t1, _ = get_centers_for_class(points_t1, class_id, flow)
            centers_t1 = centers_t1 + flow_t1

        # If no clusters are found, continue to the next class
        if clusters_t1.numel() == 0 and clusters_t2.numel() == 0:
            continue

        class_mask_t1 = points_t1[:, -2] == class_id
        class_mask_t2 = points_t2[:, -2] == class_id

        # If no clusters are found in t1, assign new ids to t2
        if clusters_t1.numel() == 0:
            for cluster_id in clusters_t2:
                mask = (class_mask_t2) & (points_t2[:, -1] == cluster_id)
                indices_t2[mask] = curr_id
                curr_id += 1
            continue

        # If no clusters are found in t2, assign ids to t1
        if clusters_t2.numel() == 0:
            for cluster_id in clusters_t1:
                mask = (class_mask_t1) & (points_t1[:, -1] == cluster_id)
                if prev_ind is None:  # if prev_ind is not provided, assign new ids
                    indices_t1[mask] = curr_id
                    curr_id += 1
                else:  # if prev_ind is provided, assign previous ids
                    indices_t1[mask] = prev_ind[mask][0]
            continue

        dists = torch.cdist(centers_t1, centers_t2)
        # associate using hungarian matching
        row_ind, col_ind = linear_sum_assignment(dists.cpu().numpy())
        used_row, used_col = set(row_ind), set(col_ind)
        for i, j in zip(row_ind, col_ind):
            mask_t1 = (class_mask_t1) & (points_t1[:, -1] == clusters_t1[i])
            mask_t2 = (class_mask_t2) & (points_t2[:, -1] == clusters_t2[j])
            if dists[i, j] > config["association"]["max_dist"]:  # threshold for association
                indices_t1[mask_t1] = (
                    curr_id if prev_ind is None else prev_ind[mask_t1][0]
                )
                curr_id += 1 if prev_ind is None else 0
                indices_t2[mask_t2] = curr_id
                curr_id += 1
            else:
                id_val = curr_id if prev_ind is None else prev_ind[mask_t1][0]
                indices_t1[mask_t1] = id_val
                indices_t2[mask_t2] = id_val
                curr_id += 1 if prev_ind is None else 0

        # Handle the case where the number of clusters in t1 and t2 are different
        if centers_t1.shape[0] > centers_t2.shape[0]:
            for i, cluster_id in enumerate(clusters_t1):
                if i in used_row:
                    continue
                mask = (class_mask_t1) & (points_t1[:, -1] == cluster_id)
                if prev_ind is None:
                    indices_t1[mask] = curr_id
                    curr_id += 1
                else:
                    indices_t1[mask] = prev_ind[mask][0]
        elif centers_t1.shape[0] < centers_t2.shape[0]:
            for j, cluster_id in enumerate(clusters_t2):
                if j in used_col:
                    continue
                mask = (class_mask_t2) & (points_t2[:, -1] == cluster_id)
                indices_t2[mask] = curr_id
                curr_id += 1
        else:
            continue

    indices_t1 = indices_t1.to(points_t1.device)
    indices_t2 = indices_t2.to(points_t2.device)
    return indices_t1, indices_t2
