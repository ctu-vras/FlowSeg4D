import os
import yaml
import argparse

import torch
import hdbscan
import numpy as np
import matplotlib.pyplot as plt

from waffleiron import Segmenter
from ScaLR.datasets import LIST_DATASETS, Collate

# from icp_flow import flow_estimation
from pan_seg_utils import transform_pointcloud, get_semantic_clustering

torch.set_default_tensor_type(torch.FloatTensor)


def get_default_parser():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument(
        "--dataset",
        type=str,
        help="Path to dataset",
        default="nuscenes",
    )
    parser.add_argument(
        "--path_dataset",
        type=str,
        help="Path to dataset",
        default="/mnt/data/Public_datasets/nuScenes/",
    )
    parser.add_argument(
        "--log_path",
        type=str,
        required=False,
        default="demo_log",
        help="Path to log folder",
    )
    parser.add_argument(
        "--restart", action="store_true", default=False, help="Restart training"
    )
    parser.add_argument(
        "--seed", default=None, type=int, help="Seed for initializing training"
    )
    parser.add_argument(
        "--gpu", default=None, type=int, help="Set to any number to use gpu 0"
    )
    parser.add_argument(
        "--multiprocessing-distributed",
        action="store_true",
        help="Use multi-processing distributed training to launch "
        "N processes per node, which has N GPUs. This is the "
        "fastest way to use PyTorch for either single node or "
        "multi node data parallel training",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="Enable autocast for mix precision training",
    )
    parser.add_argument(
        "--config_pretrain",
        type=str,
        required=False,
        default="ScaLR/configs/pretrain/WI_768_pretrain.yaml",
        help="Path to config for pretraining",
    )
    parser.add_argument(
        "--config_downstream",
        type=str,
        required=False,
        default="ScaLR/configs/downstream/nuscenes/WI_768_finetune_100p.yaml",
        help="Path to model config downstream",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        default=False,
        help="Run validation only",
    )
    parser.add_argument(
        "--pretrained_ckpt",
        type=str,
        default="ScaLR/logs/linear_probing/WI_768-DINOv2_ViT_L_14-NS_KI_PD/nuscenes/ckpt_last.pth",
        help="Path to pretrained ckpt",
    )
    parser.add_argument(
        "--linprob",
        action="store_true",
        default=False,
        help="Linear probing",
    )
    parser.add_argument("--eps", type=float, default=2.5, help="DBSCAN epsilon")
    parser.add_argument(
        "--min_points", type=int, default=15, help="DBSCAN minimum points"
    )

    return parser


def get_datasets(config, args):
    # Shared parameters
    kwargs = {
        "rootdir": args.path_dataset,
        "input_feat": config["embedding"]["input_feat"],
        "voxel_size": config["embedding"]["voxel_size"],
        "num_neighbors": config["embedding"]["neighbors"],
        "dim_proj": config["waffleiron"]["dim_proj"],
        "grids_shape": config["waffleiron"]["grids_size"],
        "fov_xyz": config["waffleiron"]["fov_xyz"],
    }

    # Get datatset
    DATASET = LIST_DATASETS.get(args.dataset.lower())
    if DATASET is None:
        raise ValueError(f"Dataset {args.dataset.lower()} not available.")

    # Train dataset
    train_dataset = DATASET(
        phase="train",
        **kwargs,
    )

    # Validation dataset
    val_dataset = DATASET(
        phase="val",
        **kwargs,
    )

    return train_dataset, val_dataset


def get_dataloader(train_dataset, val_dataset, args):
    train_sampler = None
    val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        # shuffle=(train_sampler is None),
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
        collate_fn=Collate(),
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=val_sampler,
        drop_last=False,
        collate_fn=Collate(),
    )

    return train_loader, val_loader, train_sampler


def load_model_config(file):
    with open(file, "r") as f:
        config = yaml.safe_load(f)
    return config


if __name__ == "__main__":
    parser = get_default_parser()
    args = parser.parse_args()

    # Load config files
    config = load_model_config(args.config_downstream)
    config_pretrain = load_model_config(args.config_pretrain)

    # Merge config files
    # Embeddings
    config["embedding"] = {}
    config["embedding"]["input_feat"] = config_pretrain["point_backbone"][
        "input_features"
    ]
    config["embedding"]["size_input"] = config_pretrain["point_backbone"]["size_input"]
    config["embedding"]["neighbors"] = config_pretrain["point_backbone"][
        "num_neighbors"
    ]
    config["embedding"]["voxel_size"] = config_pretrain["point_backbone"]["voxel_size"]
    # Backbone
    config["waffleiron"]["depth"] = config_pretrain["point_backbone"]["depth"]
    config["waffleiron"]["num_neighbors"] = config_pretrain["point_backbone"][
        "num_neighbors"
    ]
    config["waffleiron"]["dim_proj"] = config_pretrain["point_backbone"]["dim_proj"]
    config["waffleiron"]["nb_channels"] = config_pretrain["point_backbone"][
        "nb_channels"
    ]
    config["waffleiron"]["pretrain_dim"] = config_pretrain["point_backbone"]["nb_class"]
    config["waffleiron"]["layernorm"] = config_pretrain["point_backbone"]["layernorm"]

    # For datasets which need larger FOV for finetuning...
    if config["dataloader"].get("new_grid_shape") is not None:
        # ... overwrite config used at pretraining
        config["waffleiron"]["grids_size"] = config["dataloader"]["new_grid_shape"]
    else:
        # ... otherwise keep default value
        config["waffleiron"]["grids_size"] = config_pretrain["point_backbone"][
            "grid_shape"
        ]
    if config["dataloader"].get("new_fov") is not None:
        config["waffleiron"]["fov_xyz"] = config["dataloader"]["new_fov"]
    else:
        config["waffleiron"]["fov_xyz"] = config_pretrain["point_backbone"]["fov"]

    # --- Build network
    model = Segmenter(
        input_channels=config["embedding"]["size_input"],
        feat_channels=config["waffleiron"]["nb_channels"],
        depth=config["waffleiron"]["depth"],
        grid_shape=config["waffleiron"]["grids_size"],
        nb_class=config["classif"]["nb_class"],
        drop_path_prob=config["waffleiron"]["drop_path"],
        layer_norm=config["waffleiron"]["layernorm"],
    )

    args.batch_size = 2
    args.workers = 0

    # --- Setup ICP-Flow
    config_icp_flow = load_model_config("ICP_Flow/config.yaml")

    # --- Build nuScenes dataset
    train_dataset, val_dataset = get_datasets(config, args)
    trn_loader, val_loader, _ = get_dataloader(train_dataset, val_dataset, args)

    # Load pretrained model
    ckpt = torch.load(args.pretrained_ckpt, map_location="cpu")
    ckpt = ckpt["net"]
    new_ckpt = {}
    for k in ckpt.keys():
        if k.startswith("module"):
            new_ckpt[k[len("module.") :]] = ckpt[k]
        else:
            new_ckpt[k] = ckpt[k]

    # Adding classification layer
    model.classif = torch.nn.Conv1d(
        config["waffleiron"]["nb_channels"], config["waffleiron"]["pretrain_dim"], 1
    )

    classif = torch.nn.Conv1d(
        config["waffleiron"]["nb_channels"], config["classif"]["nb_class"], 1
    )
    torch.nn.init.constant_(classif.bias, 0)
    torch.nn.init.constant_(classif.weight, 0)
    model.classif = torch.nn.Sequential(
        torch.nn.BatchNorm1d(config["waffleiron"]["nb_channels"]),
        classif,
    )

    model.load_state_dict(new_ckpt)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()

    for i, batch in enumerate(trn_loader):
        # Network inputs
        feat = batch["feat"].to(device)
        labels = batch["labels_orig"].to(device)
        batch["upsample"] = [up.to(device) for up in batch["upsample"]]
        cell_ind = batch["cell_ind"].to(device)
        occupied_cell = batch["occupied_cells"].to(device)
        neighbors_emb = batch["neighbors_emb"].to(device)
        net_inputs = (feat, cell_ind, occupied_cell, neighbors_emb)

        # Get prediction and loss
        with torch.autocast(device):
            with torch.no_grad():
                out, tokens = model(*net_inputs)
            # Upsample to original resolution
            out_upsample = []
            for id_b, closest_point in enumerate(batch["upsample"]):
                temp = out[id_b, :, closest_point]
                out_upsample.append(temp.T)
            # Loss

        # reconstruct point clouds
        for i in range(batch["feat"].shape[0]):
            predictions = out_upsample[i].argmax(dim=1)

        # box branch
        # tokens_channels = tokens.shape[1]
        # box_reg = torch.nn.Conv1d(tokens_channels, 7, 1)  # x, y, z, l, w, h, yaw
        # box_reg = box_reg.half().to(device)  # float16

        # box_features = box_reg(tokens)
        # print("box_features - B x C x N: ", box_features.shape)

        # instance branch
        # K = 20
        # instance_reg = torch.nn.Conv1d(tokens_channels, K, 1)  # K instances
        # instance_reg = instance_reg.half().to(device)  # float16

        # instance_features = instance_reg(tokens).softmax(dim=1)
        # instance_class = instance_features.argmax(dim=1)
        # print("instance_features - B x K x N: ", instance_features.shape)

        # box to point matching (dynamic to static)
        # pcd = batch["feat"][-1, :, : out_upsample[-1].shape[0]].T.to(device)
        # pred = out_upsample[-1].argmax(dim=1)
        # pcd = torch.cat(
        #     (pcd, instance_class[-1].unsqueeze(1), pred.unsqueeze(1)), axis=1
        # )

        # unique_vals, counts = torch.unique(pred, return_counts=True)
        # max_class = unique_vals[torch.argmax(counts)]

        src_points = batch["feat"][0, :, :out_upsample[0].shape[0]].T[:, 1:4]
        src_points = src_points.to(device)
        dst_points = batch["feat"][1, :, :out_upsample[1].shape[0]].T[:, 1:4]
        dst_points = dst_points.to(device)

        # ground removal
        if False:
            # pypatchworkpp
            import pypatchworkpp

            params = pypatchworkpp.Parameters()
            grnd = pypatchworkpp.patchworkpp(params)
            grnd.estimateGround(points)
            non_ground = grnd.getNonground()
            np.save("non_ground_ppp.npy", non_ground)

            # semantic
            ground_classes = [10, 11, 12, 13]
            pred = pred.cpu().numpy()
            mask = np.isin(pred, ground_classes)
            non_ground = points[~mask][:,:3]
            np.save("non_ground_sem.npy", non_ground)

        ground_classes = torch.tensor([10, 11, 12, 13]).to(device)
        src_pred = out_upsample[0].argmax(dim=1).to(device)
        dst_pred = out_upsample[1].argmax(dim=1).to(device)

        # src_mask = torch.isin(src_pred, ground_classes)
        # dst_mask = torch.isin(dst_pred, ground_classes)

        # src_points = src_points[~src_mask]
        # dst_points = dst_points[~dst_mask]

        # Clustering
        src_points = torch.cat((src_points, src_pred.unsqueeze(1)), axis=1)
        dst_points = torch.cat((dst_points, dst_pred.unsqueeze(1)), axis=1)

        src_labels = get_semantic_clustering(src_points, config_icp_flow)
        dst_labels = get_semantic_clustering(dst_points, config_icp_flow)

        src_points = torch.cat((src_points, src_labels.unsqueeze(1)), axis=1)
        dst_points = torch.cat((dst_points, dst_labels.unsqueeze(1)), axis=1)

        # save
        src_points = src_points.cpu().numpy()
        dst_points = dst_points.cpu().numpy()
        np.save("data_src.npy", src_points)
        np.save("data_dst.npy", dst_points)

        # Ego motion
        # src_points = transform_pointcloud(src_points, batch["ego_motion"][0]["ego_motion"])
        # dst_points = transform_pointcloud(dst_points, batch["ego_motion"][1]["ego_motion"])

        # ICP-Flow
        if False:
            src_labels = torch.tensor(clustering(src_points))
            dst_labels = torch.tensor(clustering(dst_points))

            src_points = src_points.to(device)
            dst_points = dst_points.to(device)
            src_labels = src_labels.to(device)
            dst_labels = dst_labels.to(device)
            pose = torch.eye(4).to(device)
            with torch.autocast(device_type=device):
                flow = flow_estimation(config_icp_flow, src_points, dst_points, src_labels, dst_labels, pose)
             
            # save
            src_points = src_points.cpu().numpy()
            dst_points = dst_points.cpu().numpy()
            src_labels = src_labels.cpu().numpy()
            dst_labels = dst_labels.cpu().numpy()
            flow = flow.cpu().numpy()

            data_src = np.concatenate((src_points, src_labels[:, None], flow), axis=1)
            np.save("data_src.npy", data_src)
            data_dst = np.concatenate((dst_points, dst_labels[:, None]), axis=1)
            np.save("data_dst.npy", data_dst)

        break
