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
        "--rootdir", type=str, default=None, help="Path to output directory"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.rootdir is None:
        args.rootdir = args.dataroot

    nusc = NuScenes(version="v1.0-trainval", dataroot=args.dataroot, verbose=True)
    config = load_model_config("configs/config.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for scene in nusc.scene:
        print(scene["name"])
        src = nusc.get("sample", scene["first_sample_token"])
        dst = nusc.get("sample", src["next"])
        while True:
            src_points = np.fromfile(nusc.get_sample_data(src["data"]["LIDAR_TOP"])[0], dtype=np.float32)
            src_points = torch.from_numpy(src_points.reshape(-1, 5)[:, :3]).to(device)
            dst_points = np.fromfile(nusc.get_sample_data(dst["data"]["LIDAR_TOP"])[0], dtype=np.float32)
            dst_points = torch.from_numpy(dst_points.reshape(-1, 5)[:, :3]).to(device)

            panoptic_path = nusc.get("panoptic", src["data"]["LIDAR_TOP"])["filename"]
            src_labels = np.load(f"{args.rootdir}/{panoptic_path}", allow_pickle=True)["data"]
            src_labels = torch.from_numpy(src_labels.astype(np.int32) // 1000).to(device).long()

            panoptic_path = nusc.get("panoptic", dst["data"]["LIDAR_TOP"])["filename"]
            dst_labels = np.load(f"{args.rootdir}/{panoptic_path}", allow_pickle=True)["data"]
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
            np.savez_compressed(
                f"{args.rootdir}/flow/{scene['name']}_{src['token']}_{dst['token']}.npz",
                flow=flow.cpu().numpy(),
            )

            if not dst["next"]:
                break
            src = dst
            dst = nusc.get("sample", dst["next"])
