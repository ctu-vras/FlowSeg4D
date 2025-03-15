import torch

from ICP_Flow.utils_match import match_pcds


def flow_estimation(config, src_points, dst_points, src_labels, dst_labels, pose):
    assert len(src_points) == len(src_labels)

    pairs, transformations = match_pcds(
        config, src_points, dst_points, src_labels, dst_labels
    )

    T_per_point = torch.eye(4)[None, :, :].repeat(len(src_points), 1, 1).to(src_points.device)
    idxs_valid = src_labels[:, None] - pairs[:, 0][None, :]
    idxs_x, idxs_y = torch.nonzero(idxs_valid == 0, as_tuple=True)
    T_per_point[idxs_x] = transformations[idxs_y]
    T_per_point = torch.bmm(T_per_point, pose[None, :, :].expand(len(src_points), 4, 4))
    src_points_h = torch.cat(
        [src_points, src_points.new_ones(len(src_points), 1)], dim=-1
    )
    flow = torch.bmm(T_per_point, src_points_h[:, :, None])[:, 0:3, 0] - src_points

    return flow
