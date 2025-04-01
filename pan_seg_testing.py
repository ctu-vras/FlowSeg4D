import argparse

import torch
import numpy as np

from WaffleIron.waffleiron import Segmenter
from ScaLR.datasets import LIST_DATASETS, Collate

from utils.eval import EvalPQ4D
from utils.clustering import Clusterer
from utils.association import association
from utils.misc import load_model_config, transform_pointcloud

torch.set_default_tensor_type(torch.FloatTensor)


def get_default_parser():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset name",
        default="nuscenes",
    )
    parser.add_argument(
        "--path_dataset",
        type=str,
        help="Path to dataset",
        default="/mnt/data/vras/data/nuScenes-panoptic/",
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
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--verbose", action="store_true", default=False, help="Verbose debug messages")

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
        "verbose": args.verbose,
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

    args.workers = 0

    # --- Setup ICP-Flow
    config_panseg = load_model_config("configs/config.yaml")
    config_panseg["num_classes"] = config["classif"]["nb_class"]
    config_panseg["fore_classes"] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    config_panseg["ignore_classes"] = None #[10, 11, 12, 13]

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

    ind_cache = {"max_id": 0}
    prev_ind = None
    prev_scene = None
    prev_points = None

    model = model.to(device)
    model.eval()

    evaluator = EvalPQ4D(config["classif"]["nb_class"], config_panseg["ignore_classes"])
    clusterer = Clusterer(config_panseg)

    for i, batch in enumerate(trn_loader if not args.eval else val_loader):
        # network inputs
        feat = batch["feat"].to(device)
        labels = batch["labels_orig"]
        inst_lab = batch["instance_labels"]
        batch["upsample"] = [up.to(device) for up in batch["upsample"]]
        cell_ind = batch["cell_ind"].to(device)
        occupied_cell = batch["occupied_cells"].to(device)
        neighbors_emb = batch["neighbors_emb"].to(device)
        net_inputs = (feat, cell_ind, occupied_cell, neighbors_emb)

        # get semantic class prediction
        with torch.autocast(device_type=device):
            with torch.no_grad():
                out, tokens = model(*net_inputs)
            # upsample to original resolution
            out_upsample = []
            for id_b, closest_point in enumerate(batch["upsample"]):
                temp = out[id_b, :, closest_point]
                out_upsample.append(temp.T)

        # get instance prediction
        predictions = {}
        instances = {}
        batch_size = batch["feat"].shape[0]
        for src_id, dst_id in zip(range(0, batch_size - 1), range(1, batch_size)):
            src_points = batch["feat"][src_id, :, batch["upsample"][src_id]].T[:, 1:4]
            src_points = src_points.to(device)
            dst_points = batch["feat"][dst_id, :, batch["upsample"][dst_id]].T[:, 1:4]
            dst_points = dst_points.to(device)

            # ego motion
            src_points_ego = transform_pointcloud(src_points[:, :3], batch["ego"][src_id].to(device))
            dst_points_ego = transform_pointcloud(dst_points[:, :3], batch["ego"][dst_id].to(device))

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

            # semantic class
            src_pred = out_upsample[src_id].argmax(dim=1).to(device)
            dst_pred = out_upsample[dst_id].argmax(dim=1).to(device)

            # clustering
            src_points = torch.cat((src_points, src_pred.unsqueeze(1)), axis=1)
            dst_points = torch.cat((dst_points, dst_pred.unsqueeze(1)), axis=1)

            src_labels = clusterer.get_semantic_clustering(src_points)
            dst_labels = clusterer.get_semantic_clustering(dst_points)

            # scene flow
            # TODO: change for Let-It-Flow
            if False:
                src_labels = torch.tensor(clustering(src_points))
                dst_labels = torch.tensor(clustering(dst_points))

                src_points = src_points.to(device)
                dst_points = dst_points.to(device)
                src_labels = src_labels.to(device)
                dst_labels = dst_labels.to(device)
                pose = torch.eye(4).to(device)
                with torch.autocast(device_type=device):
                    flow = flow_estimation(config_panseg, src_points, dst_points, src_labels, dst_labels, pose)

            # associate -- set temporally consistent instance id
            src_points = torch.cat((src_points_ego, src_pred.unsqueeze(1), src_labels.unsqueeze(1)), axis=1)
            dst_points = torch.cat((dst_points_ego, dst_pred.unsqueeze(1), dst_labels.unsqueeze(1)), axis=1)

            ind_src, ind_dst = None, None
            if prev_ind is not None:
                if prev_scene["token"] == batch["scene"][src_id]["token"]:
                    test, ind_src = association(prev_points, src_points, config_panseg, prev_ind, ind_cache)
                    ind_cache["max_id"] = int(max(prev_ind.max(), ind_src.max()))
                    prev_ind = ind_src
                else:
                    prev_ind = None
                    ind_cache = {"max_id": 0}
            if batch["scene"][src_id]["token"] == batch["scene"][dst_id]["token"]:
                ind_src, ind_dst = association(src_points, dst_points, config_panseg, prev_ind, ind_cache)
                ind_cache["max_id"] = int(max(ind_src.max(), ind_dst.max()))
                prev_ind = ind_dst
            else:
                prev_ind = None
                ind_cache = {"max_id": 0}
            prev_points = dst_points
            prev_scene = batch["scene"][dst_id]

            if ind_src is not None and src_id not in predictions:
                predictions[src_id] = src_pred.cpu().numpy()
                instances[src_id] = ind_src.cpu().numpy()
            if ind_dst is not None:
                predictions[dst_id] = dst_pred.cpu().numpy()
                instances[dst_id] = ind_dst.cpu().numpy()

        # get ground truth and update evaluation
        start_idx = 0
        for batch_id in range(batch_size):
            end_idx = start_idx + batch["upsample"][batch_id].shape[0]
            if batch_id not in predictions:
                start_idx = end_idx
                continue

            lidarseg_labels = labels[start_idx:end_idx].cpu().numpy()
            instance_labels = inst_lab[start_idx:end_idx].cpu().numpy()
            start_idx = end_idx

            evaluator.update(
                batch["scene"][batch_id]["token"],
                predictions[batch_id],
                instances[batch_id],
                lidarseg_labels,
                instance_labels,
            )

        if (i+1) % 100 == 0 and args.verbose:
            print("\n==========================")
            print(f"Batch {i+1} done - {(i+1) * args.batch_size} samples processed")
            LSTQ, AQ_ovr, _, _, _, _, iou_mean, _, _ = evaluator.compute()
            print(f"LSTQ: {LSTQ},\nAQ_ovr: {AQ_ovr},\niou_mean: {iou_mean}")

    print("\n==========================")
    print(f"Batch {i+1} done - {(i+1) * args.batch_size} samples processed")
    LSTQ, AQ_ovr, AQ, AQ_p, AQ_r, iou, iou_mean, iou_p, iou_r = evaluator.compute()
    print(f"LSTQ: {LSTQ},\nAQ_ovr: {AQ_ovr},\nAQ: {AQ},\nAQ_p: {AQ_p},\nAQ_r: {AQ_r}")
    print(f"iou: {iou},\niou_mean: {iou_mean},\niou_p: {iou_p},\niou_r: {iou_r}")

    conf_matrix = evaluator.conf_matrix.copy()
    conf_matrix[:, evaluator.ignore] = 0
    conf_matrix[evaluator.ignore, :] = 0

    import matplotlib.pyplot as plt
    plt.imshow(conf_matrix, interpolation="nearest", cmap=plt.cm.viridis)
    plt.title("Confusion matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig("confusion_matrix.png")
    plt.close()
