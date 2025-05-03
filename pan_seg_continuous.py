import os
import time
import argparse

import torch

from WaffleIron.waffleiron import Segmenter

from utils.clustering import Clusterer
from utils.dataloaders import get_dataloader, get_datasets
from utils.association import association, long_association
from utils.misc import (
    Obj_cache,
    save_data,
    process_configs,
    print_config_cont,
    load_model_config,
    transform_pointcloud,
)


class PanSegmenter:
    def __init__(self, args):
        self.args = args

        # Load config files
        config_panseg = load_model_config("configs/config.yaml")
        config_pretrain = load_model_config(args.config_pretrain)
        config_model = load_model_config(
            config_panseg[args.dataset]["config_downstream"]
        )

        process_configs(args, config_panseg, config_pretrain, config_model)
        config_msg = print_config_cont(args, config_panseg)
        if args.save_path is not None:
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
            with open(f"{args.save_path}/config.txt", "w") as f:
                f.write(config_msg)

        self.config = config_panseg

        # Build network
        self.model = Segmenter(
            input_channels=config_model["embedding"]["size_input"],
            feat_channels=config_model["waffleiron"]["nb_channels"],
            depth=config_model["waffleiron"]["depth"],
            grid_shape=config_model["waffleiron"]["grids_size"],
            nb_class=config_model["classif"]["nb_class"],
            drop_path_prob=config_model["waffleiron"]["drop_path"],
            layer_norm=config_model["waffleiron"]["layernorm"],
        )

        # Adding classification layer
        self.model.classif = torch.nn.Conv1d(
            config_model["waffleiron"]["nb_channels"],
            config_model["waffleiron"]["pretrain_dim"],
            1,
        )
        classif = torch.nn.Conv1d(
            config_model["waffleiron"]["nb_channels"],
            config_model["classif"]["nb_class"],
            1,
        )
        torch.nn.init.constant_(classif.bias, 0)
        torch.nn.init.constant_(classif.weight, 0)
        self.model.classif = torch.nn.Sequential(
            torch.nn.BatchNorm1d(config_model["waffleiron"]["nb_channels"]),
            classif,
        )

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
        self.device = torch.device(device)

        # Set model to evaluation mode
        self.model.load_state_dict(new_ckpt)
        self.model = self.model.to(device)
        self.model.compile()
        self.model.eval()

        # Initialize
        self.prev_ind = None
        self.prev_scene = None
        self.prev_points = None
        self.clusterer = Clusterer(config_panseg)
        self.obj_cache = Obj_cache(config_model["classif"]["nb_class"])

    def __call__(self, data):
        times = [time.time()]

        # network inputs
        feat = data["feat"].to(self.device)
        cell_ind = data["cell_ind"].to(self.device)
        occupied_cell = data["occupied_cells"].to(self.device)
        neighbors_emb = data["neighbors_emb"].to(self.device)
        net_inputs = (feat, cell_ind, occupied_cell, neighbors_emb)
        times.append(time.time())

        # get semantic class prediction
        with torch.inference_mode():
            out, tokens = self.model(*net_inputs)
        out = out[0].argmax(dim=0)
        times.append(time.time())

        # upsample to original resolution
        data["upsample"] = data["upsample"][0].to(self.device)
        out_upsample = out[data["upsample"]]
        times.append(time.time())

        # get instance prediction
        src_points = feat[0, 1:4, data["upsample"]].T
        src_features = tokens[0, :, data["upsample"]].T

        # ego motion compensation
        src_points_ego = transform_pointcloud(
            src_points, data["ego"][0].to(self.device)
        )

        # get semantic class
        src_pred = out_upsample.unsqueeze(1)

        # clustering
        src_points = torch.cat((src_points, src_pred), axis=1)
        src_labels = self.clusterer.get_semantic_clustering(src_points)

        # create data - ego compensated xyz + features + semantic class + cluster id
        src_points = torch.cat(
            (src_points_ego, src_features, src_pred, src_labels.unsqueeze(1)), axis=1
        )
        times.append(time.time())

        # associate -- set temporally consistent instance id
        ind_src = None
        if (
            self.prev_scene is None
            or not self.prev_scene["token"] == data["scene"][0]["token"]
        ):
            self.prev_ind = None
            self.obj_cache.reset()
            self.prev_points = torch.zeros_like(src_points)

        if self.config["association"]["use_long"]:
            _, ind_src = long_association(
                self.prev_points,
                src_points,
                self.config,
                self.prev_ind,
                self.obj_cache,
                None,
            )
        else:
            _, ind_src = association(
                self.prev_points,
                src_points,
                self.config,
                self.prev_ind,
                self.obj_cache,
                None,
            )
        self.prev_ind = ind_src
        self.obj_cache.max_id = int(max(self.prev_ind.max(), ind_src.max()))

        self.prev_points = src_points
        self.prev_scene = data["scene"][0]
        times.append(time.time())

        if args.verbose:
            print(
                f"Total time: {times[5] - times[0]:.2f} s\n"
                f"  SemSeg data prep: {times[1] - times[0]:.2f} | "
                f"Semantic segmentation: {times[2] - times[1]:.2f} | "
                f"Upsample: {times[3] - times[2]:.2f} | "
                f"InsSeg data prep: {times[4] - times[3]:.2f} | "
                f"Instance segmentation: {times[5] - times[4]:.2f} | "
            )

        # save segmentation files
        if args.save_path is not None:
            save_data(
                args.save_path,
                data["scene"][0]["name"],
                data["filename"][0],
                src_pred.cpu().numpy().squeeze(),
                ind_src.cpu().numpy(),
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
        "--gpu", default=None, type=int, help="Set to a number of gpu to use"
    )
    parser.add_argument(
        "--save_path", type=str, default=None, help="Path to save segmentation files"
    )
    parser.add_argument(
        "--clustering", type=str, default=None, help="Clustering method"
    )
    parser.add_argument(
        "--short",
        action="store_true",
        default=False,
        help="Do not use long association",
    )
    parser.add_argument(
        "--verbose", action="store_true", default=False, help="Verbose mode"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.workers = 0

    # Set parameters -- necessary for the dataloader
    # not needed in real deployment
    args.eval = True
    args.test = False
    args.flow = False
    args.batch_size = 1

    # Load config files
    config_panseg = load_model_config("configs/config.yaml")
    config_pretrain = load_model_config(args.config_pretrain)
    config_model = load_model_config(config_panseg[args.dataset]["config_downstream"])
    process_configs(args, config_panseg, config_pretrain, config_model)

    # Load dataset
    dataset = get_datasets(config_model, args)
    dataloader = get_dataloader(dataset, args)

    # Initialize segmenter
    segmenter = PanSegmenter(args)

    try:
        for i, batch in enumerate(dataloader):
            segmenter(batch)
    except KeyboardInterrupt:
        print("Keyboard interrupt, exiting...")
    except Exception as e:
        raise e
