import os
import copy
import argparse
from dataclasses import dataclass
from typing import Union, Tuple, Optional

import yaml
import torch
import numpy as np


###############################
# IO functions
###############################

def load_model_config(file: str) -> dict:
    """
    Load the model configuration from a YAML file.
    Args:
        file (str): The path to the YAML file.
    Returns:
        dict: The loaded configuration.
    """
    with open(file, "r") as f:
        config = yaml.safe_load(f)
    return config

def save_data(
        save_path: str, scene_name: str, filename: str, semantic: torch.Tensor, instance: torch.Tensor
) -> None:
    """
    Save the data to a file.
    Args:
        save_path (str): The path to save the data.
        scene_name (str): The name of the scene.
        filename (str): The path to the original file.
        semantic (torch.Tensor): The semantic labels.
        instance (torch.Tensor): The instance labels.
    """
    save_dir = os.path.join(save_path, scene_name, "predictions")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    filename = filename.split("/")[-1].split(".")[0] + ".label"
    save_file = os.path.join(save_dir, filename)
    save_data = (instance.astype(np.uint32) << 16) | semantic.astype(np.uint32)

    save_data.tofile(save_file)

###############################
# Config functions
###############################

def print_config(args: argparse.Namespace, config: dict) -> str:
    """
    Print the configuration settings.
    Args:
        args (argparse.Namespace): The arguments passed to the script.
        config (dict): The configuration settings.
    Returns:
        str: The formatted configuration string.
    """
    msg = ""
    msg += f"Dataset: {args.dataset}\n"
    msg += f"  path: {args.path_dataset}\n"
    msg += f"  split: {'eval' if args.eval else 'test' if args.test else 'train'}\n"
    msg += f"  foreground classes: {config[f'{args.dataset}']['fore_classes']}\n"

    if args.save_path is not None:
        msg += f"Save path: {args.save_path}\n"

    msg += f"Batch size: {args.batch_size}\n"

    msg += f"GPU: {torch.cuda.is_available()}\n"
    if args.gpu:
        msg += f"  GPU ID: {args.gpu}\n"

    msg += f"Use flow: {args.flow}\n"
    msg += f"Use gt: {args.use_gt}\n"

    clustering_method = config["clustering"]["clustering_method"] if args.clustering is None else args.clustering
    msg += f"Clustering: {clustering_method}\n"
    msg += f"  max number of clusters: {config['clustering']['num_clusters']}\n"
    if clustering_method == "hdbscan":
        msg += f"  min_samples: {config['clustering']['min_cluster_size']}\n"
    elif clustering_method == "dbscan":
        msg += f"  epsilon: {config['clustering']['epsilon']}\n"
        msg += f"  min_samples: {config['clustering']['min_cluster_size']}\n"
    elif clustering_method == "alpine":
        msg += f"  margin: {config['alpine']['margin']}\n"
        msg += f"  neighbours: {config['alpine']['neighbours']}\n"
        source = config["alpine"]["bbox_source"]
        msg += f"  bbox source: {source}\n"
        msg += f"  bbox: {config[f'{args.dataset}'][f'bbox_{source}']}\n"

    msg += f"Association:\n"
    msg += f"  max distance: {config['association']['max_dist']}\n"
    if not args.short or (args.short and config['association']['use_feat']):
        msg += f"  max feature distance: {config['association']['max_feat']}\n"
    if not args.short:
        msg += f"  life: {config['association']['life']}\n"

    msg += f"Checkpoint: {args.pretrained_ckpt}\n"

    msg += f"Verbose: {args.verbose}\n"

    print("\nConfiguration:")
    print("=" * 20)
    print(msg)
    return msg

def print_config_cont(args: argparse.Namespace, config: dict) -> str:
    """
    Print the configuration settings for the continuous run of panoptic
    segmentation algorithm.
    Args:
        args (argparse.Namespace): The arguments passed to the script.
        config (dict): The configuration settings.
    Returns:
        str: The formatted configuration string.
    """
    msg = ""

    clustering_method = config["clustering"]["clustering_method"] if args.clustering is None else args.clustering
    msg += f"Clustering: {clustering_method}\n"
    msg += f"  max number of clusters: {config['clustering']['num_clusters']}\n"
    if clustering_method == "hdbscan":
        msg += f"  min_samples: {config['clustering']['min_cluster_size']}\n"
    elif clustering_method == "dbscan":
        msg += f"  epsilon: {config['clustering']['epsilon']}\n"
        msg += f"  min_samples: {config['clustering']['min_cluster_size']}\n"
    elif clustering_method == "alpine":
        msg += f"  margin: {config['alpine']['margin']}\n"
        msg += f"  neighbours: {config['alpine']['neighbours']}\n"
        source = config["alpine"]["bbox_source"]
        msg += f"  bbox source: {source}\n"
        msg += f"  bbox: {config[f'{args.dataset}'][f'bbox_{source}']}\n"

    msg += f"Association:\n"
    msg += f"  max distance: {config['association']['max_dist']}\n"
    if not args.short or (args.short and config['association']['use_feat']):
        msg += f"  max feature distance: {config['association']['max_feat']}\n"
    if not args.short:
        msg += f"  life: {config['association']['life']}\n"

    if args.save_path is not None:
        msg += f"Save path: {args.save_path}\n"

    msg += f"GPU: {torch.cuda.is_available()}\n"
    if args.gpu:
        msg += f"  GPU ID: {args.gpu}\n"

    msg += f"Checkpoint: {args.pretrained_ckpt}\n"

    msg += f"Verbose: {args.verbose}\n"

    print("\nConfiguration:")
    print("=" * 20)
    print(msg)
    return msg

def process_configs(
        args: argparse.Namespace, config_panseg: dict, config_pretrain: dict, config_model: dict
) -> None:
    """
    Process the config files and merge different configurations.
    Args:
        args (argparse.Namespace): The arguments passed to the script.
        config_panseg (dict): The config file for panseg.
        config_pretrain (dict): The config file for pretraining.
        config_model (dict): The config file for the model.
    """
    config_panseg["num_classes"] = config_model["classif"]["nb_class"]
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

    # Merge config files
    # Embeddings
    config_model["embedding"] = {}
    config_model["embedding"]["input_feat"] = config_pretrain["point_backbone"][
        "input_features"
    ]
    config_model["embedding"]["size_input"] = config_pretrain["point_backbone"]["size_input"]
    config_model["embedding"]["neighbors"] = config_pretrain["point_backbone"][
        "num_neighbors"
    ]
    config_model["embedding"]["voxel_size"] = config_pretrain["point_backbone"]["voxel_size"]

    # Backbone
    config_model["waffleiron"]["depth"] = config_pretrain["point_backbone"]["depth"]
    config_model["waffleiron"]["num_neighbors"] = config_pretrain["point_backbone"][
        "num_neighbors"
    ]
    config_model["waffleiron"]["dim_proj"] = config_pretrain["point_backbone"]["dim_proj"]
    config_model["waffleiron"]["nb_channels"] = config_pretrain["point_backbone"][
        "nb_channels"
    ]
    config_model["waffleiron"]["pretrain_dim"] = config_pretrain["point_backbone"]["nb_class"]
    config_model["waffleiron"]["layernorm"] = config_pretrain["point_backbone"]["layernorm"]

    # For datasets which need larger FOV for finetuning...
    if config_model["dataloader"].get("new_grid_shape") is not None:
        # ... overwrite config used at pretraining
        config_model["waffleiron"]["grids_size"] = config_model["dataloader"]["new_grid_shape"]
    else:
        # ... otherwise keep default value
        config_model["waffleiron"]["grids_size"] = config_pretrain["point_backbone"][
            "grid_shape"
        ]
    if config_model["dataloader"].get("new_fov") is not None:
        config_model["waffleiron"]["fov_xyz"] = config_model["dataloader"]["new_fov"]
    else:
        config_model["waffleiron"]["fov_xyz"] = config_pretrain["point_backbone"]["fov"]

###############################
# Transformations
###############################

def transform_pointcloud(
    points: torch.Tensor, transform_matrix: Union[torch.Tensor, np.ndarray]
) -> torch.Tensor:
    """
    Returns the transformed pointcloud given the transformation matrix.

    Args:
        points (torch.Tensor): The input pointcloud of shape (N, 3)
        transform_matrix (torch.Tensor or np.ndarray): The transformation matrix of shape (4, 4)

    Returns:
        torch.Tensor: The transformed pointcloud of shape (N, 3)
    """
    assert points.dim() == 2, "The input pointcloud should have shape (N, M)"
    assert points.shape[1] == 3, "The input pointcloud should have shape (N, 3)"

    if not isinstance(transform_matrix, torch.Tensor):
        transform_matrix = torch.tensor(
            transform_matrix, dtype=points.dtype, device=points.device
        )

    points_tr = torch.cat(
        (points, torch.ones((points.shape[0], 1), device=points.device)), axis=1
    )
    points_tr = torch.mm(transform_matrix, points_tr.T).T

    return points_tr[:, :3]

def get_centers_for_class(
    points: torch.Tensor,
    class_id: int,
    feat: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes cluster centers (median) for a given class. If feat is provided, it computes
    the median for the feature tensor instead of the original points.

    Args:
        points (torch.Tensor): Input tensor of shape (N, D), where last two columns
                               represent class ID and cluster ID.
        class_id (int): The class ID to filter clusters for.
        feat (Optional[torch.Tensor]): Optional feature tensor of shape (N, M) to compute
                                       centers for. It could be flow - shape (N, 3), or
                                       some per point features - shape (N, M).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - Tensor of shape (num_clusters, 3 or M) containing computed median.
            - Tensor of unique cluster IDs.
    """
    class_mask = points[:, -2] == class_id
    clusters = torch.unique(points[class_mask, -1]).long()
    clusters = clusters[clusters != -1].sort()[0]

    if clusters.numel() == 0:
        return torch.empty(0, 3), clusters

    if feat is None:
        centers = torch.stack(
            [
                points[(class_mask) & (points[:, -1] == cluster_id), :3].median(dim=0).values
                for cluster_id in clusters
            ]
        )
    else:
        if feat.shape[1] == 3:
            centers = torch.stack(
                [
                    feat[(class_mask) & (points[:, -1] == cluster_id)].median(dim=0).values
                    for cluster_id in clusters
                ]
            )
        else:
            centers = torch.stack(
                [
                    feat[(class_mask) & (points[:, -1] == cluster_id)].mean(dim=0)
                    for cluster_id in clusters
                ]
            )

    return centers.double(), clusters

###############################
# Data classes
###############################

@dataclass
class Obj_cache:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.max_id = 0
        self.prev_instances = [dict() for _ in range(self.num_classes)]

    def add_instance(self, class_id, instance):
        self.prev_instances[class_id][instance.id] = copy.deepcopy(instance)

    def del_instance(self, class_id, instance_id):
        if instance_id in self.prev_instances[class_id].keys():
            del self.prev_instances[class_id][instance_id]
        else:
            print(f"Instance {instance_id} not found in class {class_id}.")

    def update_step(self):
        for i, instance_dict in enumerate(self.prev_instances):
            keys_to_delete = []
            for key, obj in instance_dict.items():
                obj.life -= 1
                if obj.life < 0:
                    keys_to_delete.append(key)
            for key in keys_to_delete:
                self.del_instance(i, key)

@dataclass
class Instance_data:
    id: int
    cl_id: int
    life: int
    center: torch.Tensor
    # bbox : torch.Tensor
    feature: torch.Tensor

    def __repr__(self):
        return f"Instance_data(id={self.id}, center={self.center}, life={self.life})"
