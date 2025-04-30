import argparse

import torch

from WaffleIron.waffleiron import Segmenter
from ScaLR.datasets import LIST_DATASETS, Collate

from utils.eval import EvalPQ4D
from utils.clustering import Clusterer
from utils.association import association, long_association
from utils.misc import Obj_cache, load_model_config, transform_pointcloud, print_config

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
        "--gpu", default=None, type=int, help="Set to a number of gpu to use"
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
        default="ScaLR/configs/downstream/nuscenes/WI_768_linprob.yaml",
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
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--verbose", action="store_true", default=False, help="Verbose debug messages")
    parser.add_argument("--clustering", type=str, default=None, help="Clustering method")
    parser.add_argument("--flow", action="store_true", default=False, help="Use flow estimation")
    parser.add_argument(
        "--use_gt",
        action="store_true",
        default=False,
        help="Use ground truth labels for semantic segmentation",
    )
    parser.add_argument("--short", action="store_true", default=False, help="Do not use long association")

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

    # --- Setup 
    config_panseg = load_model_config("configs/config.yaml")

    # Load config files
    config = load_model_config(config_panseg[args.dataset]["config_downstream"])
    config_pretrain = load_model_config(args.config_pretrain)

    config_panseg["num_classes"] = config["classif"]["nb_class"]
    config_panseg["fore_classes"] = config_panseg[args.dataset]["fore_classes"]
    config_panseg["ignore_classes"] = None
    if args.clustering is not None:
        config_panseg["clustering"]["clustering_method"] = args.clustering.lower()
    if config_panseg["clustering"]["clustering_method"] == "alpine":
        config_panseg["alpine"]["BBOX_WEB"] = config_panseg[args.dataset]["bbox_web"]
        config_panseg["alpine"]["BBOX_DATASET"] = config_panseg[args.dataset]["bbox_dataset"]
    if args.short:
        config_panseg["association"]["use_long"] = False
    else:
        config_panseg["association"]["use_long"] = True

    print_config(args, config_panseg)

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

    device = "cpu"
    if torch.cuda.is_available():
        if args.gpu is not None:
            device = f"cuda:{args.gpu}"
        else:
            device = "cuda"
    device = torch.device(device)

    ind_cache = Obj_cache(config["classif"]["nb_class"])
    prev_ind = None
    prev_scene = None
    prev_sample = None
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
        scene_flow = batch["scene_flow"].to(device)
        batch["upsample"] = [up.to(device) for up in batch["upsample"]]
        cell_ind = batch["cell_ind"].to(device)
        occupied_cell = batch["occupied_cells"].to(device)
        neighbors_emb = batch["neighbors_emb"].to(device)
        net_inputs = (feat, cell_ind, occupied_cell, neighbors_emb)

        # get semantic class prediction
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
        s_idx = 0
        for src_id, dst_id in zip(range(0, batch_size - 1), range(1, batch_size)):
            src_points = feat[src_id, :, batch["upsample"][src_id]].T[:, 1:4]
            src_points = src_points.to(device)
            src_features = tokens[src_id, :, batch["upsample"][src_id]].T
            dst_points = feat[dst_id, :, batch["upsample"][dst_id]].T[:, 1:4]
            dst_points = dst_points.to(device)
            dst_features = tokens[dst_id, :, batch["upsample"][dst_id]].T

            if args.flow:
                flow = scene_flow[src_id, :, batch["upsample"][src_id]].T
            else:
                flow = None
            
            # ego motion
            src_points_ego = transform_pointcloud(src_points, batch["ego"][src_id].to(device))
            dst_points_ego = transform_pointcloud(dst_points, batch["ego"][dst_id].to(device))

            # semantic class
            if not args.use_gt:
                src_pred = out_upsample[src_id].argmax(dim=1).to(device)
                dst_pred = out_upsample[dst_id].argmax(dim=1).to(device)
            else:
                e_idx = s_idx + batch["upsample"][src_id].shape[0]
                src_pred = labels[s_idx:e_idx].to(device)
                dst_pred = labels[e_idx:e_idx + batch["upsample"][dst_id].shape[0]].to(device)
                s_idx = e_idx

            # ground removal
            ground_classes = torch.tensor([10, 11, 12, 13], device=device)
            src_mask = torch.isin(src_pred, ground_classes)
            src_non_ground = src_points[~src_mask]
            dst_mask = torch.isin(dst_pred, ground_classes)
            dst_non_ground = dst_points[~dst_mask]

            # clustering
            src_points = torch.cat((src_points, src_pred.unsqueeze(1)), axis=1)
            dst_points = torch.cat((dst_points, dst_pred.unsqueeze(1)), axis=1)

            src_labels = clusterer.get_semantic_clustering(src_points)
            dst_labels = clusterer.get_semantic_clustering(dst_points)

            # create data - ego compensated xyz + features + semantic class + cluster id
            src_points = torch.cat((src_points_ego, src_features, src_pred.unsqueeze(1), src_labels.unsqueeze(1)), axis=1)
            dst_points = torch.cat((dst_points_ego, dst_features, dst_pred.unsqueeze(1), dst_labels.unsqueeze(1)), axis=1)

            # associate -- set temporally consistent instance id
            ind_src, ind_dst = None, None
            if prev_ind is not None and src_id == 0:
                if prev_scene["token"] == batch["scene"][src_id]["token"]:
                    if config_panseg["association"]["use_long"]:
                        _, ind_src = long_association(prev_points, src_points, config_panseg, prev_ind, ind_cache, prev_flow)
                    else:
                        _, ind_src = association(prev_points, src_points, config_panseg, prev_ind, ind_cache, prev_flow)
                    ind_cache.max_id = int(max(prev_ind.max(), ind_src.max()))
                    prev_ind = ind_src
                else:
                    prev_ind = None
                    ind_cache.reset()
            if batch["scene"][src_id]["token"] == batch["scene"][dst_id]["token"]:
                if config_panseg["association"]["use_long"]:
                    ind_src, ind_dst = long_association(src_points, dst_points, config_panseg, prev_ind, ind_cache, flow)
                else:
                    ind_src, ind_dst = association(src_points, dst_points, config_panseg, prev_ind, ind_cache, flow)
                ind_cache.max_id = int(max(ind_src.max(), ind_dst.max()))
                prev_ind = ind_dst
            else:
                prev_ind = None
                ind_cache.reset()
            prev_points = dst_points
            prev_scene = batch["scene"][dst_id]
            prev_sample = batch["sample"][dst_id]
            if args.flow:
                prev_flow = scene_flow[dst_id, :, batch["upsample"][dst_id]].T
            else:
                prev_flow = None

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

    with open(f"results/{config_panseg['clustering']['clustering_method']}.out", "w") as fh:
        fh.write(f"LSTQ: {LSTQ},\nAQ_ovr: {AQ_ovr},\nAQ: {AQ},\nAQ_p: {AQ_p},\nAQ_r: {AQ_r}\n")
        fh.write(f"iou: {iou},\niou_mean: {iou_mean},\niou_p: {iou_p},\niou_r: {iou_r}\n")

    conf_matrix = evaluator.conf_matrix.copy()
    conf_matrix[:, evaluator.ignore] = 0
    conf_matrix[evaluator.ignore, :] = 0
