import os
import argparse

import torch
import numpy as np
from nuscenes.nuscenes import NuScenes

from utils.misc import load_model_config
from utils.flow import flow_estimation_lif


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute flow")
    parser.add_argument(
        "--dataroot", type=str, help="Path to NuScenes dataset", required=True
    )
    parser.add_argument(
        "--savedir", type=str, default=None, help="Path to output directory"
    )
    parser.add_argument(
        "--dataset", type=str, default="nuscenes", help="Dataset name"
    )
    parser.add_argument(
        "--restart", type=int, default=None, help="Restart from specified scene number"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.savedir is None:
        args.savedir = args.dataroot

    config = load_model_config("configs/config.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.dataset == "nuscenes":
        nusc = NuScenes(version="v1.0-trainval", dataroot=args.dataroot, verbose=True)

        for scene in nusc.scene:
            if args.restart is not None:
                scene_index = int(scene["name"].split("-")[-1])
                if scene_index < args.restart:
                    continue
            print(f"Processing scene {scene['name']}")
            src = nusc.get("sample", scene["first_sample_token"])
            dst = nusc.get("sample", src["next"])
            while True:
                src_points = np.fromfile(nusc.get_sample_data(src["data"]["LIDAR_TOP"])[0], dtype=np.float32)
                src_points = torch.from_numpy(src_points.reshape(-1, 5)[:, :3]).to(device)
                dst_points = np.fromfile(nusc.get_sample_data(dst["data"]["LIDAR_TOP"])[0], dtype=np.float32)
                dst_points = torch.from_numpy(dst_points.reshape(-1, 5)[:, :3]).to(device)

                panoptic_path = nusc.get("panoptic", src["data"]["LIDAR_TOP"])["filename"]
                src_labels = np.load(f"{args.dataroot}/{panoptic_path}", allow_pickle=True)["data"]
                src_labels = torch.from_numpy(src_labels.astype(np.int32) // 1000).to(device).long()

                panoptic_path = nusc.get("panoptic", dst["data"]["LIDAR_TOP"])["filename"]
                dst_labels = np.load(f"{args.dataroot}/{panoptic_path}", allow_pickle=True)["data"]
                dst_labels = torch.from_numpy(dst_labels.astype(np.int32) // 1000).to(device).long()

                flow = flow_estimation_lif(
                    config=config,
                    src_points=src_points,
                    dst_points=dst_points,
                    src_labels=src_labels,
                    dst_labels=dst_labels,
                    device=device,
                )

                # Save flow to disk or process it as needed
                filename = f"{scene['name']}_{src['token']}_{dst['token']}.npz"
                np.savez_compressed(
                    os.path.join(args.savedir, "flow", filename),
                    flow=flow.cpu().numpy(),
                )

                if not dst["next"]:
                    break
                src = dst
                dst = nusc.get("sample", dst["next"])

    elif args.dataset == "semantic_kitti":
        for scene in range(22):
            if args.restart is not None:
                if scene < args.restart:
                    continue
            print(f"Processing scene {scene:02d}")
            scene_dir = os.path.join(args.dataroot, f"dataset/sequences/{scene:02d}")

            for i in range(len(os.listdir(os.path.join(scene_dir, "velodyne"))) - 1):
                # Load source and destination point clouds
                src_points = np.fromfile(os.path.join(scene_dir, "velodyne", f"{i:06d}.bin"), dtype=np.float32)
                src_points = torch.from_numpy(src_points.reshape(-1, 4)[:, :3]).to(device)
                dst_points = np.fromfile(os.path.join(scene_dir, "velodyne", f"{i+1:06d}.bin"), dtype=np.float32)
                dst_points = torch.from_numpy(dst_points.reshape(-1, 4)[:, :3]).to(device)

                # Load labels
                src_labels = np.fromfile(
                    os.path.join(scene_dir, "labels", f"{i:06d}.label"),
                    dtype=np.uint32,
                )
                src_labels = src_labels & 0xFFFF
                src_labels = torch.from_numpy(src_labels.astype(np.int32)).to(device).long()
                dst_labels = np.fromfile(
                    os.path.join(scene_dir, "labels", f"{i+1:06d}.label"),
                    dtype=np.uint32,
                )
                dst_labels = dst_labels & 0xFFFF
                dst_labels = torch.from_numpy(dst_labels.astype(np.int32)).to(device).long()

                flow = flow_estimation_lif(
                    config=config,
                    src_points=src_points,
                    dst_points=dst_points,
                    src_labels=src_labels,
                    dst_labels=dst_labels,
                    device=device,
                )

                # Save flow to disk or process it as needed
                filename = f"{scene:02d}_{i:06d}_{i+1:06}.npz"
                np.savez_compressed(
                    os.path.join(args.savedir, "dataset/flow", filename),
                    flow=flow.cpu().numpy(),
                )

    else:
        raise ValueError(f"Dataset {args.dataset} not supported.")
