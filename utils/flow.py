import torch
from pytorch3d.ops.knn import knn_points

from LetItFlow import let_it_flow
from ICP_Flow.utils_match import match_pcds


def flow_estimation_icp(config, src_points, dst_points, src_labels, dst_labels, pose):
    # TODO: Sometimes fails, fix?
    assert len(src_points) == len(src_labels)

    pairs, transformations = match_pcds(
        config, src_points, dst_points, src_labels, dst_labels
    )

    T_per_point = (
        torch.eye(4)[None, :, :].repeat(len(src_points), 1, 1).to(src_points.device)
    )
    idxs_valid = src_labels[:, None] - pairs[:, 0][None, :]
    idxs_x, idxs_y = torch.nonzero(idxs_valid == 0, as_tuple=True)
    T_per_point[idxs_x] = transformations[idxs_y]
    T_per_point = torch.bmm(T_per_point, pose[None, :, :].expand(len(src_points), 4, 4))
    src_points_h = torch.cat(
        [src_points, src_points.new_ones(len(src_points), 1)], dim=-1
    )
    flow = torch.bmm(T_per_point, src_points_h[:, :, None])[:, 0:3, 0] - src_points

    return flow


def flow_estimation_lif(config, src_points, dst_points, src_labels, dst_labels, device):
    p1, p2 = src_points.unsqueeze(0), dst_points.unsqueeze(0)
    c1, c2 = src_labels, dst_labels
    f1 = torch.zeros(p1.shape, device=device, requires_grad=True)

    optimizer = torch.optim.Adam([f1], lr=config["let_it_flow"]["lr"])
    RigidLoss = let_it_flow.SC2_KNN_cluster_aware(p1, K=config["let_it_flow"]["K"], d_thre=config["let_it_flow"]["d_thre"])

    for i in range(config["let_it_flow"]["iters"]):
        loss = 0

        dist, nn, _ = knn_points(p1 + f1, p2, lengths1=None, lengths2=None, K=1, return_nn=True)
        dist_b, _, _ = knn_points(p2, p1 + f1, lengths1=None, lengths2=None, K=1, return_nn=True)
        loss += config["let_it_flow"]["dist_w"] * \
                (dist[dist < config["let_it_flow"]["trunc_dist"]].mean() + dist_b[dist_b < config["let_it_flow"]["trunc_dist"]].mean())

        sc_loss = RigidLoss(f1, c1)

        if config["let_it_flow"]["sc_w"] > 0:
            loss += config["let_it_flow"]["sc_w"] * sc_loss
            loss += config["let_it_flow"]["sc_w"] + let_it_flow.center_rigidity_loss(p1, f1, c1[None] + 1)

        loss += f1[..., 2].norm().mean()

        if i % 10 == 0 and config["let_it_flow"]["passing_ids"]:
            c1 = let_it_flow.pass_id_clusters(c1, c2, nn)

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    return f1.detach().squeeze()
