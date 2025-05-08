import os

import torch
import numpy as np
from pytorch3d.ops.knn import knn_points

from LetItFlow import let_it_flow


def flow_estimation_lif(config, src_points, dst_points, src_labels, dst_labels, device):
    p1, p2 = src_points.unsqueeze(0), dst_points.unsqueeze(0)
    c1, c2 = src_labels, dst_labels
    f1 = torch.zeros(p1.shape, device=device, requires_grad=True)

    optimizer = torch.optim.Adam([f1], lr=config["lr"])
    RigidLoss = let_it_flow.SC2_KNN_cluster_aware(
        p1, K=config["K"], d_thre=config["d_thre"]
    )

    for i in range(config["iters"]):
        loss = 0

        dist, nn, _ = knn_points(
            p1 + f1, p2, lengths1=None, lengths2=None, K=1, return_nn=True
        )
        dist_b, _, _ = knn_points(
            p2, p1 + f1, lengths1=None, lengths2=None, K=1, return_nn=True
        )
        loss += config["dist_w"] * (
            dist[dist < config["trunc_dist"]].mean()
            + dist_b[dist_b < config["trunc_dist"]].mean()
        )

        sc_loss = RigidLoss(f1, c1)

        if config["sc_w"] > 0:
            loss += config["sc_w"] * sc_loss
            loss += config["sc_w"] + let_it_flow.center_rigidity_loss(
                p1, f1, c1[None] + 1
            )

        loss += f1[..., 2].norm().mean()

        if i % 10 == 0 and config["passing_ids"]:
            c1 = let_it_flow.pass_id_clusters(c1, c2, nn)

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    return f1.detach().squeeze()


def load_flow(args, scene, src_info, dst_info):
    filename = (
        scene["name"]
        + "_"
        + str(src_info["token"])
        + "_"
        + str(dst_info["token"])
        + ".npz"
    )
    if args.dataset == "nuscenes":
        file_path = os.path.join(args.path_dataset, "flow", filename)
    elif args.dataset == "semantic_kitti":
        file_path = os.path.join(args.path_dataset, "dataset/flow", filename)

    flow = np.load(file_path, allow_pickle=True)["flow"]

    return torch.from_numpy(flow).float()
