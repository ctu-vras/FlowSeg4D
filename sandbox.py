#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import torch
import numpy as np

from ground_removal import GroundRemoval
from utils import load_pcd, visualize_pcd, get_clusters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, help="Path to the root data directory")
    parser.add_argument("-f", "--file", type=str, help="Path to PCD file")
    parser.add_argument(
        "--dataset",
        type=str,
        default="av2",
        required=True,
        help="Name of the dataset, default: av2",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        help="Path to save the results, if not specified, the results will be saved in <data_dir>_xyz",
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Visualize the results"
    )
    parser.add_argument("--verbose", action="store_true", help="Print verbose messages")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.file is None and args.data_root is None:
        raise ValueError("Either --file or --data_root must be provided")
    elif args.file is not None and args.data_root is not None:
        raise ValueError("Only one of --file or --data_root must be provided")

    ground_removal = GroundRemoval(args, standalone=True)
    if args.file is not None:
        pcd = load_pcd(args.file, args.dataset)
        pcd, _ = ground_removal.run_individual_scan(pcd)
        labels = get_clusters(pcd, eps=0.5, min_points=50)
        if args.visualize:
            visualize_pcd(pcd, labels)
