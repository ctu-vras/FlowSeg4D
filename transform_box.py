#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transform box")

    parser.add_argument("-d", "--debug", action="store_true", help="Debug mode")

    return parser.parse_args()


def vis_box_bev(box: torch.Tensor, color: str) -> None:
    corners = np.array(
        [
            [-box[3] / 2, -box[4] / 2],
            [-box[3] / 2, box[4] / 2],
            [box[3] / 2, box[4] / 2],
            [box[3] / 2, -box[4] / 2],
        ]
    )
    rot_mat = np.array(
        [
            [np.cos(box[6]), -np.sin(box[6])],
            [np.sin(box[6]), np.cos(box[6])],
        ]
    )
    corners @= rot_mat.T
    corners += np.array([box[0], box[1]])
    corners = np.vstack([corners, corners[0]])
    plt.plot(corners[:, 0], corners[:, 1], color=color)


def get_box_transform(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    transform = torch.eye(4)
    transform[:3, 3] = box2[:3] - box1[:3]
    transform[:3, :3] = torch.tensor(
        [
            [torch.cos(box2[6] - box1[6]), -torch.sin(box2[6] - box1[6]), 0],
            [torch.sin(box2[6] - box1[6]), torch.cos(box2[6] - box1[6]), 0],
            [0, 0, 1],
        ]
    )

    return transform


def transform_box(box: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    box_t = torch.zeros_like(box)
    box_t[:3] = torch.matmul(transform[:3, :3], box[:3]) + transform[:3, 3]
    box_t[3:5] = box[3:5]
    box_t[6] = box[6] + torch.atan2(transform[1, 0], transform[0, 0])

    return box_t


def main(args: argparse.Namespace) -> None:
    box1 = torch.tensor([0, 0, 0, 2, 4, 2, 0], dtype=torch.float32)
    box2 = torch.tensor([2.3, 2, 0, 2, 4, 2, -np.pi / 6], dtype=torch.float32)

    fig = plt.figure()
    vis_box_bev(box1, "r-")
    vis_box_bev(box2, "r-")
    transform = get_box_transform(box1, box2)
    vis_box_bev(transform_box(box1, transform), "b--")
    plt.axis("equal")
    plt.show()


if __name__ == "__main__":
    args = parse_args()

    main(args)
