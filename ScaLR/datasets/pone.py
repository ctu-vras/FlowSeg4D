import os

import numpy as np
from .pc_dataset import PCDataset

# For normalizing intensities
MEAN_INT = 0.391358
STD_INT = 0.151813


class PoneSemSeg(PCDataset):
    CLASS_NAME = [  # List of class names -- use the same as nuScenes
        "barrier",  # 0
        "bicycle",  # 1
        "bus",  # 2
        "car",  # 3
        "construction_vehicle",  # 4
        "motorcycle",  # 5
        "pedestrian",  # 6
        "traffic_cone",  # 7
        "trailer",  # 8
        "truck",  # 9
        "driveable_surface",  # 10
        "other_flat",  # 11
        "sidewalk",  # 12
        "terrain",  # 13
        "manmade",  # 14
        "vegetation",  # 15
    ]

    def __init__(self, ratio="100p", **kwargs):
        super().__init__(**kwargs)

        # For normalizing intensities
        self.mean_int = MEAN_INT
        self.std_int = STD_INT

        current_folder = os.path.dirname(os.path.realpath(__file__))

        # Load the list of frames
        self.ratio = ratio
        if self.phase == "train":
            self.list_frames = np.load(
                os.path.join(current_folder, "list_frames_pone.npz"), allow_pickle=True
            )["train"]
        elif self.phase == "val":
            self.list_frames = np.load(
                os.path.join(current_folder, "list_frames_pone.npz"), allow_pickle=True
            )["val"]
        elif self.phase == "test":
            self.list_frames = np.load(
                os.path.join(current_folder, "list_frames_pone.npz"), allow_pickle=True
            )["test"]
        else:
            raise ValueError(f"Unknown phase {self.phase}")

    def __len__(self):
        return len(self.list_frames)

    def load_pc(self, index):
        # Load point cloud
        pc = np.load(
            os.path.join(self.rootdir, self.list_frames[index][0]), allow_pickle=True
        )["pcd"]

        # Load segmentation labels
        labels = np.zeros(pc.shape[0], dtype=np.int32)

        # Label 0 should be ignored
        labels = labels - 1
        labels[labels == -1] = 255

        return pc, labels, self.list_frames[index][2]

    def get_ego_motion(self, index):
        data = np.load(
            os.path.join(self.rootdir, self.list_frames[index][0]), allow_pickle=True
        )["odom"]

        scene_name = self.list_frames[index][0].split("/")[-1][:-9]
        scene = {"name": scene_name, "token": scene_name}

        return data["transformation"], scene, self.list_frames[index][1]

    def get_panoptic_labels(self, index):
        return None, None

    def get_scene_flow(self, index):
        return None
