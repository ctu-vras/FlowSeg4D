# Copyright 2024 - Valeo Comfort and Driving Assistance - valeo.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import numpy as np


class Compose:
    def __init__(self, transformations):
        self.transformations = transformations

    def __call__(self, pcloud, labels, instance, flow):
        for t in self.transformations:
            pcloud, labels, instance, flow = t(pcloud, labels, instance, flow)
        return pcloud, labels, instance, flow


class RandomApply:
    def __init__(self, transformation, prob=0.5):
        self.prob = prob
        self.transformation = transformation

    def __call__(self, pcloud, labels, instance, flow):
        if torch.rand(1) < self.prob:
            pcloud, labels, instance, flow = self.transformation(pcloud, labels, instance, flow)
        return pcloud, labels, instance, flow


class Transformation:
    def __init__(self, inplace=False):
        self.inplace = inplace

    def __call__(self, pcloud, labels, instance, flow):
        if labels is None:
            return (
                (pcloud, None, None, None) if self.inplace else (np.array(pcloud, copy=True), None, None, None)
            )

        out = (
            (pcloud, labels, instance, flow)
            if self.inplace
            else (np.array(pcloud, copy=True), np.array(labels, copy=True), np.array(instance, copy=True), np.array(flow, copy=True))
        )
        return out


class Identity(Transformation):
    def __init__(self, inplace=False):
        super().__init__(inplace)

    def __call__(self, pcloud, labels, instance, flow):
        return super().__call__(pcloud, labels, instance, flow)


class Rotation(Transformation):
    def __init__(self, dim=2, range=np.pi, inplace=False):
        super().__init__(inplace)
        self.range = range
        self.inplace = inplace
        if dim == 2:
            self.dims = (0, 1)
        elif dim == 1:
            self.dims = (0, 2)
        elif dim == 0:
            self.dims = (1, 2)
        elif dim == 6:
            self.dims = (4, 5)

    def __call__(self, pcloud, labels, instance, flow):
        # Build rotation matrix
        theta = (2 * torch.rand(1)[0] - 1) * self.range
        # Build rotation matrix
        rot = np.array(
            [
                [np.cos(theta), np.sin(theta)],
                [-np.sin(theta), np.cos(theta)],
            ]
        )
        # Apply rotation
        pcloud, labels, instance, flow = super().__call__(pcloud, labels, instance, flow)
        pcloud[:, self.dims] = pcloud[:, self.dims] @ rot
        return pcloud, labels, instance, flow


class Scale(Transformation):
    def __init__(self, dims=(0, 1), range=0.05, inplace=False):
        super().__init__(inplace)
        self.dims = dims
        self.range = range

    def __call__(self, pcloud, labels, instance, flow):
        pcloud, labels, instance, flow = super().__call__(pcloud, labels, instance, flow)
        scale = 1 + (2 * torch.rand(1).item() - 1) * self.range
        pcloud[:, self.dims] *= scale
        return pcloud, labels, instance, flow


class FlipXY(Transformation):
    def __init__(self, inplace=False):
        super().__init__(inplace=inplace)

    def __call__(self, pcloud, labels, instance, flow):
        pcloud, labels, instance, flow = super().__call__(pcloud, labels, instance, flow)
        id = torch.randint(2, (1,))[0]
        pcloud[:, id] *= -1.0
        return pcloud, labels, instance, flow


class LimitNumPoints(Transformation):
    def __init__(self, dims=(0, 1, 2), max_point=30000, random=False):
        super().__init__(inplace=True)
        self.dims = dims
        self.max_points = max_point
        self.random = random
        assert max_point > 0

    def __call__(self, pcloud, labels, instance, flow, return_idx=False):
        pc, labels, instance, flow = super().__call__(pcloud, labels, instance, flow)
        if pc.shape[0] > self.max_points:
            if self.random:
                center = torch.randint(pc.shape[0], (1,))[0]
                center = pc[center : center + 1, self.dims]
            else:
                center = np.zeros((1, len(self.dims)))
            idx = np.argsort(np.square(pc[:, self.dims] - center).sum(axis=1))[
                : self.max_points
            ]
            pc = pc[idx]
            labels = None if labels is None else labels[idx]
            instance = None if instance is None else instance[idx]
            flow = None if flow is None else flow[idx]
        else:
            idx = np.arange(pc.shape[0])
        if return_idx:
            return pc, labels, instance, flow, idx
        else:
            return pc, labels, instance, flow


class Crop(Transformation):
    def __init__(self, dims=(0, 1, 2), fov=((-5, -5, -5), (5, 5, 5)), eps=1e-4):
        super().__init__(inplace=True)
        self.dims = dims
        self.fov = fov
        self.eps = eps
        assert len(fov[0]) == len(fov[1]), "Min and Max FOV must have the same length."
        for i, (min, max) in enumerate(zip(*fov)):
            assert (
                min < max
            ), f"Field of view: min ({min}) < max ({max}) is expected on dimension {i}."

    def __call__(self, pcloud, labels, instance, flow, return_mask=False):
        pc, labels, instance, flow = super().__call__(pcloud, labels, instance, flow)

        where = None
        for i, d in enumerate(self.dims):
            temp = (pc[:, i] > self.fov[0][i] + self.eps) & (
                pc[:, i] < self.fov[1][i] - self.eps
            )
            where = temp if where is None else where & temp

        if return_mask:
            return pc[where], None if labels is None else labels[where], None if instance is None else instance[where], None if flow is None else flow[where], where
        else:
            return pc[where], None if labels is None else labels[where], None if instance is None else instance[where], None if flow is None else flow[where]


class Voxelize(Transformation):
    def __init__(self, dims=(0, 1, 2), voxel_size=0.1, random=False):
        super().__init__(inplace=True)
        self.dims = dims
        self.voxel_size = voxel_size
        self.random = random
        assert voxel_size >= 0

    def __call__(self, pcloud, labels, instance, flow):
        pc, labels, instance, flow = super().__call__(pcloud, labels, instance, flow)
        if self.voxel_size <= 0:
            return pc, labels, instance, flow

        if self.random:
            permute = torch.randperm(pc.shape[0])
            pc, labels = pc[permute], None if labels is None else labels[permute]
            instance = None if instance is None else instance[permute]
            flow = None if flow is None else flow[permute]

        pc_shift = pc[:, self.dims] - pc[:, self.dims].min(0, keepdims=True)

        _, ind = np.unique(
            (pc_shift / self.voxel_size).astype("int"), return_index=True, axis=0
        )

        return pc[ind, :], None if labels is None else labels[ind], None if instance is None else instance[ind], None if flow is None else flow[ind]
