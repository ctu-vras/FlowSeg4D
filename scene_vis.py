import os
import argparse

import numpy as np

from utils.misc import load_config
from utils.visualization import visualize_scene


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize point cloud and labels.")
    parser.add_argument(
        "--pcd_dir",
        type=str,
        required=True,
        help="Directory containing point cloud files.",
    )
    parser.add_argument(
        "--labels_dir",
        type=str,
        required=True,
        help="Directory containing label files.",
    )
    parser.add_argument(
        "--instances",
        action="store_true",
        help="Flag to indicate if instance labels are used.",
    )
    parser.add_argument(
        "--dataset", type=str, default="pone", help="What dataset config to use"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.pcd_dir):
        raise FileNotFoundError(f"Point cloud directory {args.pcd_dir} does not exist.")
    if not os.path.exists(args.labels_dir):
        raise FileNotFoundError(f"Labels directory {args.labels_dir} does not exist.")

    if args.dataset.lower() not in ["pone", "nuscenes"]:
        raise ValueError(f"Dateset {args.dataset} not supported.")
    config = load_config("configs/visualization.yaml")[args.dataset.lower()]
    config["instances"] = args.instances
    config["colors"] = np.array(config["colors"])

    try:
        visualize_scene(config, args.pcd_dir, args.labels_dir)
    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Exiting...")
    except Exception as e:
        raise e
