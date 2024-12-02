#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sklearn

import argparse

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# from ground_removal import GroundRemoval
from transform_box import vis_box_bev
from utils import visualize_pcd, get_clusters, quaternion_to_yaw, load_boxes
from utils import get_points_in_box, load_nuScenes, load_pcd
from point_fitting import icp_transform, good_match


INTEREST_CLASSES_AV2 = ["BOX_TRUCK", "BUS", "LARGE_VEHICLE", "REGULAR_VEHICLE"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, help="Path to the root data directory")
    parser.add_argument("-fp", "--file_pcd", type=str, help="Path to PCD file")
    parser.add_argument("-fb", "--file_box", type=str, help="Path to PCD file")
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
    parser.add_argument("--eps", type=float, default=0.7, help="DBSCAN epsilon")
    parser.add_argument(
        "--min_points", type=int, default=10, help="DBSCAN minimum points"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.file_pcd is None and args.data_root is None:
        raise ValueError("Either --file or --data_root must be provided")
    elif args.file_pcd is not None and args.data_root is not None:
        raise ValueError("Only one of --file or --data_root must be provided")

    # ground_removal = GroundRemoval(args, standalone=True)
    if args.dataset == "nuscenes":
        if args.data_root is None:
            raise ValueError("For nuScenes --data_root must be provided")
    else:
        if args.file_pcd is not None:
            # pcd, _ = ground_removal.run_individual_file(args.file_pcd)
            pcd = load_pcd(args.file_pcd, args.dataset)
            labels = get_clusters(pcd, eps=args.eps, min_points=args.min_points)
            if args.visualize:
                visualize_pcd(pcd, labels)

            if args.file_box is not None:
                ego_box = torch.tensor([0, 0, 0, 4.841, 1.834, 1.445, 0])
                boxes = load_boxes(args.file_box, args.dataset)
                timestamp = boxes.iloc[0]["timestamp_ns"]

                labels = boxes["category"].unique()
                labels = np.append(labels, "EGO_VEHICLE")
                colors = plt.get_cmap("tab20")(np.linspace(0, 1, len(labels)))
                legend_handles = [
                    mpatches.Patch(color=colors[i], label=labels[i])
                    for i in range(len(labels))
                ]

                for i in range(boxes.shape[0] - 1):
                    box = boxes.iloc[i]
                    box2 = boxes.iloc[i + 1]
                    if (
                        box["category"] not in INTEREST_CLASSES_AV2
                        or box2["category"] not in INTEREST_CLASSES_AV2
                    ):
                        continue
                    if timestamp != box["timestamp_ns"]:
                        timestamp = box["timestamp_ns"]
                        # vis_box_bev(ego_box, colors[-1])
                        # plt.title(f"Timestamp: {timestamp}")
                        # plt.legend(handles=legend_handles)
                        # plt.scatter(pcd[:, 0], pcd[:, 1], s=1, c="black")
                        # plt.show()
                        exit()
                    box_r = torch.tensor(
                        [
                            box["tx_m"],
                            box["ty_m"],
                            box["tz_m"],
                            box["length_m"],
                            box["width_m"],
                            box["height_m"],
                            quaternion_to_yaw(
                                torch.tensor(
                                    [box["qw"], box["qx"], box["qy"], box["qz"]]
                                )
                            ),
                        ]
                    )
                    box_r_2 = torch.tensor(
                        [
                            box2["tx_m"],
                            box2["ty_m"],
                            box2["tz_m"],
                            box2["length_m"],
                            box2["width_m"],
                            box2["height_m"],
                            quaternion_to_yaw(
                                torch.tensor(
                                    [box2["qw"], box2["qx"], box2["qy"], box2["qz"]]
                                )
                            ),
                        ]
                    )
                    object_pcd = get_points_in_box(pcd, box_r)
                    object_pcd_2 = get_points_in_box(pcd, box_r_2)
                    icp_res = icp_transform(object_pcd, object_pcd_2)
                    good_match(object_pcd, object_pcd_2, icp_res)
                    # if box["category"] in INTEREST_CLASSES_AV2:
                    #     object_pcd = get_points_in_box(pcd, box_r)
                    #     print(f"Object type: {box['category']}")
                    #     if object_pcd.shape[0] != 0:
                    #         visualize_pcd(object_pcd, None)
                    # label = box["category"]
                    # color = colors[np.where(labels == label)[0][0]]
                    # vis_box_bev(box_r, color)

                # plt.show()
