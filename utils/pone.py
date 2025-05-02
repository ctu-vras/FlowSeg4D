import os
import argparse

import tqdm
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="PONE", help="Dataset name")
    parser.add_argument(
        "--path_dataset",
        type=str,
        default="/mnt/personal/vlkjan6/PONE",
        help="Path to the dataset",
    )
    parser.add_argument(
        "--intensity",
        action="store_true",
        help="Calculate standard deviation and mean of intensity"
    )

    return parser.parse_args()


def process_raw_dataset(args):
    if not args.dataset.lower() == "pone":
        raise NotImplementedError(f"Dataset {args.dataset} is not supported yet")

    dataset_path = args.path_dataset
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path {dataset_path} does not exist")

    data_split = ["train", "val", "test"]
    data_split_dir = [["3"], ["11"], ["4"]]

    list_frame = [[] for _ in range(len(data_split))]

    for split_id in range(len(data_split)):
        print(f"Processing {data_split[split_id]} split")
        for split_dir in tqdm.tqdm(data_split_dir[split_id]):
            split_dir = os.path.join(dataset_path, split_dir)
            if not os.path.exists(split_dir):
                raise FileNotFoundError(f"Split directory {split_dir} does not exist")
            for scene_dir in tqdm.tqdm(sorted(os.listdir(split_dir))):
                if os.path.isdir(os.path.join(split_dir, scene_dir)):
                    process_scene(args, os.path.join(split_dir, scene_dir), list_frame[split_id], data_split[split_id])
        print(f"Processing completed for {data_split[split_id]} split found {len(list_frame[split_id])} frames\n")

    # Save the list_frame to a file
    np.savez(
        os.path.join(dataset_path, "list_frames_pone.npz"),
        train=list_frame[0],
        val=list_frame[1],
        test=list_frame[2],
    )

def process_scene(args, scene_dir, list_frame, split_name):
    scene_name = scene_dir.split("/")[-1]
    scene_data = np.load(os.path.join(scene_dir, (scene_name + "_PCD.npz")), allow_pickle=True)
    scan_data = scene_data["scan_list"]
    odom_data = scene_data["odom_list"]

    for i in range(scan_data.shape[0]):
        scan = scan_data[i]
        odom = odom_data[i]

        # Convert the scan data to a point cloud format
        pcd = np.concatenate([scan["x"], scan["z"].reshape(-1, 1), scan["i"].reshape(-1, 1)], axis=1)

        # Save the scan and odom data
        filename = scene_name + f"_{i:04d}.npz"
        save_path = os.path.join(args.path_dataset, split_name, filename)
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        np.savez(
            save_path,
            pcd=pcd,
            odom=odom,
        )

        # Append the frame to the list
        list_frame.append((os.path.join(split_name, filename), i, filename))

def calculate_intensity(args):
    if not args.dataset.lower() == "pone":
        raise NotImplementedError(f"Dataset {args.dataset} is not supported yet")

    dataset_path = args.path_dataset
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path {dataset_path} does not exist")

    data_split = ["train", "val", "test"]
    data_split_dir = [["3"], ["11"], ["4"]]

    intensity_values = np.array([])

    for split_id in range(len(data_split)):
        print(f"Processing {data_split[split_id]} split")
        for split_dir in data_split_dir[split_id]:
            split_dir = os.path.join(dataset_path, split_dir)
            if not os.path.exists(split_dir):
                raise FileNotFoundError(f"Split directory {split_dir} does not exist")
            for scene_dir in tqdm.tqdm(sorted(os.listdir(split_dir))):
                if os.path.isdir(os.path.join(split_dir, scene_dir)):
                    process_scene_intensity(os.path.join(split_dir, scene_dir), intensity_values)

def process_scene_intensity(scene_dir, intensity_values):
    scene_name = scene_dir.split("/")[-1]
    scene_data = np.load(os.path.join(scene_dir, (scene_name + "_PCD.npz")), allow_pickle=True)
    scan_data = scene_data["scan_list"]

    for i in range(scan_data.shape[0]):
        intensity_values = np.concatenate([intensity_values, scan_data[i]["i"].reshape(-1)])

    # Calculate mean and standard deviation
    mean_intensity = np.mean(intensity_values)
    std_intensity = np.std(intensity_values)

    print(f"Mean Intensity: {mean_intensity}, Std Intensity: {std_intensity}")


if __name__ == "__main__":
    args = parse_args()
    if not args.intensity:
        process_raw_dataset(args)
    else:
        calculate_intensity(args)
