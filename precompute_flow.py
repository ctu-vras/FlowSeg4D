import os
import argparse
from functools import reduce

import torch
import numpy as np
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix

from utils.flow import flow_estimation_lif
from utils.misc import load_model_config, transform_pointcloud

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
torch.set_num_threads(4)


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
    parser.add_argument(
        "--gpu", default=None, type=int, help="Set to a number of gpu to use"
    )

    return parser.parse_args()


def get_ego_motion_nuscenes(nusc, sample):
    scene = nusc.get('scene', sample['scene_token'])
    ref_sample = nusc.get('sample', scene['first_sample_token'])

    ref_sd_rec = nusc.get('sample_data', ref_sample['data']['LIDAR_TOP'])
    ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
    ref_cs_rec = nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])

    ref_from_car = transform_matrix(ref_cs_rec['translation'], Quaternion(ref_cs_rec['rotation']), inverse=True)
    car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']),
                                       inverse=True)

    cur_sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    current_pose_rec = nusc.get('ego_pose', cur_sd_rec['ego_pose_token'])
    current_cs_rec = nusc.get('calibrated_sensor', cur_sd_rec['calibrated_sensor_token'])

    global_from_car = transform_matrix(current_pose_rec['translation'],
                                       Quaternion(current_pose_rec['rotation']), inverse=False)
    car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),
                                        inverse=False)

    ego_motion = reduce(np.dot, [ref_from_car, car_from_global, global_from_car, car_from_current])

    return ego_motion


if __name__ == "__main__":
    args = parse_args()
    if args.savedir is None:
        args.savedir = args.dataroot

    config = load_model_config("configs/config.yaml")
    device = "cpu"
    if torch.cuda.is_available():
        if args.gpu is not None:
            device = f"cuda:{args.gpu}"
        else:
            device = "cuda"
    device = torch.device(device)

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

                src_ego = get_ego_motion_nuscenes(nusc, src)
                dst_ego = get_ego_motion_nuscenes(nusc, dst)

                src_points = transform_pointcloud(src_points, src_ego)
                dst_points = transform_pointcloud(dst_points, dst_ego)

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
            poses = np.loadtxt(os.path.join(scene_dir, "poses.txt")).reshape(-1, 3, 4)
            poses_h = np.zeros((poses.shape[0], 4, 4))
            poses_h[:, :3, :] = poses
            poses_h[:, 3, 3] = 1
            poses_h = torch.from_numpy(poses_h).to(device).double()

            pose_o = torch.linalg.inv(poses_h[0])

            for i in range(len(os.listdir(os.path.join(scene_dir, "velodyne"))) - 1):
                # Load source and destination point clouds
                src_points = np.fromfile(os.path.join(scene_dir, "velodyne", f"{i:06d}.bin"), dtype=np.float32)
                src_points = torch.from_numpy(src_points.reshape(-1, 4)[:, :3]).to(device).double()
                dst_points = np.fromfile(os.path.join(scene_dir, "velodyne", f"{i+1:06d}.bin"), dtype=np.float32)
                dst_points = torch.from_numpy(dst_points.reshape(-1, 4)[:, :3]).to(device).double()

                src_ego = pose_o @ poses_h[i]
                dst_ego = pose_o @ poses_h[i + 1]

                src_points = transform_pointcloud(src_points, src_ego)
                dst_points = transform_pointcloud(dst_points, dst_ego)

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
