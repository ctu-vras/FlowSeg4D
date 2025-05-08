import os
import time
import argparse

import torch
import numpy as np

from WaffleIron.waffleiron import Segmenter

from utils.eval import EvalPQ4D
from utils.clustering import Clusterer
from utils.dataloaders import get_dataloader, get_datasets
from utils.association import association, long_association
from utils.misc import (
    Obj_cache,
    save_data,
    print_config,
    process_configs,
    load_config,
    transform_pointcloud,
)


def parse_args():
    parser = argparse.ArgumentParser(description="4D Panoptic Segmentation")
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
        "--pretrained_ckpt",
        type=str,
        default="ScaLR/logs/linear_probing/WI_768-DINOv2_ViT_L_14-NS_KI_PD/nuscenes/ckpt_last.pth",
        help="Path to pretrained ckpt",
    )
    parser.add_argument(
        "--use_gt",
        action="store_true",
        default=False,
        help="Use ground truth labels for semantic segmentation",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        default=False,
        help="Run validation split of dataset",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        default=False,
        help="Run testing split of dataset",
    )
    parser.add_argument(
        "--gpu", default=None, type=int, help="Set to a number of gpu to use"
    )
    parser.add_argument(
        "--save_path", type=str, default=None, help="Path to save segmentation files"
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument(
        "--verbose", action="store_true", default=False, help="Verbose debug messages"
    )
    parser.add_argument(
        "--clustering", type=str, default=None, help="Clustering method"
    )
    parser.add_argument(
        "--flow", action="store_true", default=False, help="Use flow estimation"
    )
    parser.add_argument(
        "--short",
        action="store_true",
        default=False,
        help="Do not use long association",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.workers = 0
    if args.batch_size < 1:
        raise ValueError("Batch size must be greater than 0")
    elif args.batch_size == 1:
        raise ValueError(
            "Batch size of 1 is not supported, for batch size of 1 use pan_seg_continuous.py"
        )

    # Load config files
    config_panseg = load_config("configs/config.yaml")
    config_pretrain = load_config(args.config_pretrain)
    config_model = load_config(config_panseg[args.dataset]["config_downstream"])

    process_configs(args, config_panseg, config_pretrain, config_model)
    config_msg = print_config(args, config_panseg)
    if args.save_path is not None:
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        with open(f"{args.save_path}/config.txt", "w") as f:
            f.write(config_msg)

    # Build network
    model = Segmenter(
        input_channels=config_model["embedding"]["size_input"],
        feat_channels=config_model["waffleiron"]["nb_channels"],
        depth=config_model["waffleiron"]["depth"],
        grid_shape=config_model["waffleiron"]["grids_size"],
        nb_class=config_model["classif"]["nb_class"],
        drop_path_prob=config_model["waffleiron"]["drop_path"],
        layer_norm=config_model["waffleiron"]["layernorm"],
    )

    # Adding classification layer
    classif = torch.nn.Conv1d(
        config_model["waffleiron"]["nb_channels"],
        config_model["classif"]["nb_class"],
        1,
    )
    torch.nn.init.constant_(classif.bias, 0)
    torch.nn.init.constant_(classif.weight, 0)
    model.classif = torch.nn.Sequential(
        torch.nn.BatchNorm1d(config_model["waffleiron"]["nb_channels"]),
        classif,
    )

    # Load dataset
    dataset = get_datasets(config_model, args)
    dataloader = get_dataloader(dataset, args)

    # Load pretrained model
    ckpt = torch.load(args.pretrained_ckpt, map_location="cpu", weights_only=True)
    ckpt = ckpt["net"]
    new_ckpt = {}
    for k in ckpt.keys():
        if k.startswith("module"):
            new_ckpt[k[len("module.") :]] = ckpt[k]
        else:
            new_ckpt[k] = ckpt[k]

    # Set device
    device = "cpu"
    if torch.cuda.is_available():
        if args.gpu is not None:
            device = f"cuda:{args.gpu}"
        else:
            device = "cuda"
    device = torch.device(device)

    # Set model to evaluation mode
    model.load_state_dict(new_ckpt)
    model = model.to(device)
    model.compile()
    model.eval()

    # Initialize
    prev_ind = None
    prev_scene = None
    prev_points = None
    clusterer = Clusterer(config_panseg)
    ind_cache = Obj_cache(config_model["classif"]["nb_class"])
    evaluator = EvalPQ4D(
        config_model["classif"]["nb_class"], config_panseg["ignore_classes"]
    )

    # For SemanticKITTI initialize inverse mapping
    if args.dataset == "semantic_kitti":
        sem_kitti = load_config("configs/semantic-kitti.yaml")
        mapper = np.vectorize(sem_kitti["learning_map_inv"].__getitem__)

    for i, batch in enumerate(dataloader):
        # network inputs
        feat = batch["feat"].to(device)
        cell_ind = batch["cell_ind"].to(device)
        occupied_cell = batch["occupied_cells"].to(device)
        neighbors_emb = batch["neighbors_emb"].to(device)
        net_inputs = (feat, cell_ind, occupied_cell, neighbors_emb)

        # other variables
        labels = batch["labels_orig"]
        inst_lab = batch["instance_labels"]
        scene_flow = batch["scene_flow"].to(device)

        # get semantic class prediction
        with torch.inference_mode():
            out, tokens = model(*net_inputs)

        # upsample to original resolution
        out_upsample = []
        batch["upsample"] = [up.to(device) for up in batch["upsample"]]
        for id_b, closest_point in enumerate(batch["upsample"]):
            temp = out[id_b, :, closest_point]
            out_upsample.append(temp.T)

        # get instance prediction
        s_idx = 0
        instances = {}
        predictions = {}
        batch_size = batch["feat"].shape[0]
        for src_id, dst_id in zip(range(0, batch_size - 1), range(1, batch_size)):
            src_points = feat[src_id, 1:4, batch["upsample"][src_id]].T
            dst_points = feat[dst_id, 1:4, batch["upsample"][dst_id]].T

            src_features = tokens[src_id, :, batch["upsample"][src_id]].T
            dst_features = tokens[dst_id, :, batch["upsample"][dst_id]].T

            if args.flow:
                flow = scene_flow[src_id, :, batch["upsample"][src_id]].T
            else:
                flow = None

            # ego motion compensation
            src_points_ego = transform_pointcloud(
                src_points, batch["ego"][src_id].to(device)
            )
            dst_points_ego = transform_pointcloud(
                dst_points, batch["ego"][dst_id].to(device)
            )

            # get semantic class
            if not args.use_gt:
                src_pred = out_upsample[src_id].argmax(dim=1).unsqueeze(1)
                dst_pred = out_upsample[dst_id].argmax(dim=1).unsqueeze(1)
            else:
                e_idx = s_idx + batch["upsample"][src_id].shape[0]
                src_pred = labels[s_idx:e_idx].to(device)
                dst_pred = labels[
                    e_idx : e_idx + batch["upsample"][dst_id].shape[0]
                ].to(device)
                s_idx = e_idx

            # clustering
            src_points = torch.cat((src_points, src_pred), axis=1)
            dst_points = torch.cat((dst_points, dst_pred), axis=1)

            src_labels = clusterer.get_semantic_clustering(src_points)
            dst_labels = clusterer.get_semantic_clustering(dst_points)

            # create data - ego compensated xyz + features + semantic class + cluster id
            src_points = torch.cat(
                (src_points_ego, src_features, src_pred, src_labels.unsqueeze(1)),
                axis=1,
            )
            dst_points = torch.cat(
                (dst_points_ego, dst_features, dst_pred, dst_labels.unsqueeze(1)),
                axis=1,
            )

            # associate -- set temporally consistent instance id
            ind_src, ind_dst = None, None
            if prev_scene is not None and src_id == 0:
                if prev_scene["token"] == batch["scene"][src_id]["token"]:
                    if config_panseg["association"]["use_long"]:
                        prev_ind, ind_src = long_association(
                            prev_points,
                            src_points,
                            config_panseg,
                            prev_ind,
                            ind_cache,
                            prev_flow,
                        )
                    else:
                        prev_ind, ind_src = association(
                            prev_points,
                            src_points,
                            config_panseg,
                            prev_ind,
                            ind_cache,
                            prev_flow,
                        )

                    # save segmentation files
                    if args.save_path is not None:
                        preds = prev_points[:, -2].cpu().numpy()
                        if args.dataset == "semantic_kitti":
                            preds = mapper(preds + 1)
                        save_data(
                            args.save_path,
                            prev_scene["name"],
                            prev_filename,
                            preds,
                            prev_ind.cpu().numpy(),
                        )
                    ind_cache.max_id = int(max(ind_cache.max_id, prev_ind.max(), ind_src.max()))
                    prev_ind = ind_src
                else:
                    prev_ind = None
                    ind_cache.reset()
            if batch["scene"][src_id]["token"] == batch["scene"][dst_id]["token"]:
                if config_panseg["association"]["use_long"]:
                    ind_src, ind_dst = long_association(
                        src_points, dst_points, config_panseg, prev_ind, ind_cache, flow
                    )
                else:
                    ind_src, ind_dst = association(
                        src_points, dst_points, config_panseg, prev_ind, ind_cache, flow
                    )
                ind_cache.max_id = int(max(ind_cache.max_id, ind_src.max(), ind_dst.max()))
                prev_ind = ind_dst
            else:
                prev_ind = None
                ind_cache.reset()
            prev_points = dst_points
            prev_scene = batch["scene"][dst_id]
            prev_filename = batch["filename"][dst_id]
            if args.flow:
                prev_flow = scene_flow[dst_id, :, batch["upsample"][dst_id]].T
            else:
                prev_flow = None

            if ind_src is not None and src_id not in predictions:
                predictions[src_id] = src_pred.cpu().numpy().squeeze(1)
                instances[src_id] = ind_src.cpu().numpy()
            if ind_dst is not None:
                predictions[dst_id] = dst_pred.cpu().numpy().squeeze(1)
                instances[dst_id] = ind_dst.cpu().numpy()

        # get ground truth and update evaluation
        start_idx = 0
        for batch_id in range(batch_size):
            end_idx = start_idx + batch["upsample"][batch_id].shape[0]
            if batch_id not in predictions:
                start_idx = end_idx
                continue

            if not args.test:
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

            # save segmentation files
            if args.save_path is not None:
                if args.dataset == "semantic_kitti":
                    predictions[batch_id] = mapper(predictions[batch_id] + 1)
                save_data(
                    args.save_path,
                    batch["scene"][batch_id]["name"],
                    batch["filename"][batch_id],
                    predictions[batch_id],
                    instances[batch_id],
                )

        if (i + 1) % 100 == 0 and args.verbose:
            print("\n==========================")
            print(f"Batch {i+1} done - {(i+1) * args.batch_size} samples processed")
            LSTQ, AQ_ovr, _, _, _, _, iou_mean, _, _ = evaluator.compute()
            print(f"LSTQ: {LSTQ},\nAQ_ovr: {AQ_ovr},\niou_mean: {iou_mean}")

    print("\n==========================")
    print(f"Batch {i+1} done - {(i+1) * args.batch_size} samples processed")
    LSTQ, AQ_ovr, AQ, AQ_p, AQ_r, iou, iou_mean, iou_p, iou_r = evaluator.compute()
    print(f"LSTQ: {LSTQ},\nAQ_ovr: {AQ_ovr},\nAQ: {AQ},\nAQ_p: {AQ_p},\nAQ_r: {AQ_r}")
    print(f"iou: {iou},\niou_mean: {iou_mean},\niou_p: {iou_p},\niou_r: {iou_r}")

    if np.isnan(LSTQ):
        exit()

    with open(
        time.strftime("results/Log_%Y-%m-%d_%H-%M-%S.out", time.gmtime()), "w"
    ) as fh:
        fh.write(f"Config:\n{config_msg}\n\n")
        fh.write(
            f"LSTQ: {LSTQ},\nAQ_ovr: {AQ_ovr},\nAQ: {AQ},\nAQ_p: {AQ_p},\nAQ_r: {AQ_r}\n"
        )
        fh.write(
            f"iou: {iou},\niou_mean: {iou_mean},\niou_p: {iou_p},\niou_r: {iou_r}\n"
        )
