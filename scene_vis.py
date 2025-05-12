import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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
        "--dataset", type=str, default="pone", help="What dataset config to use"
    )
    parser.add_argument(
        "--instances",
        action="store_true",
        help="Flag to indicate if instance labels are used.",
    )
    parser.add_argument(
        "--fps", type=int, default=None, help="Frames per second for visualization."
    )
    return parser.parse_args()


def show_legend(colors, labels):
    _, ax = plt.subplots()
    for i, name, color in zip(range(len(colors)), labels, colors):
        ax.text(0.16, 0.9 - i * 0.05, name, transform=ax.transAxes)
        ax.add_patch(
            mpatches.Rectangle(
                (0.1, 0.9 - i * 0.05), 0.05, 0.05, color=color, transform=ax.transAxes
            )
        )

    ax.axis("off")
    plt.pause(0.1)


if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.pcd_dir):
        raise FileNotFoundError(f"Point cloud directory {args.pcd_dir} does not exist.")
    if not os.path.exists(args.labels_dir):
        raise FileNotFoundError(f"Labels directory {args.labels_dir} does not exist.")

    if args.dataset.lower() not in ["pone", "nuscenes", "semantic_kitti"]:
        raise ValueError(f"Dateset {args.dataset} not supported.")
    config = load_config("configs/visualization.yaml")[args.dataset.lower()]
    config["dataset"] = args.dataset.lower()
    config["instances"] = args.instances
    if config["colors"] is not None:
        config["colors"] = np.array(config["colors"]) / 255
    if args.fps is not None:
        config["fps"] = args.fps

    if not config["instances"] and config["colors"] is not None:
        show_legend(config["colors"], config["names"])

    try:
        visualize_scene(config, args.pcd_dir, args.labels_dir)
    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Exiting...")
    except Exception as e:
        raise e
