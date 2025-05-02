import argparse

import torch

from ScaLR.datasets import LIST_DATASETS, Collate


def get_datasets(config: dict, args: argparse.Namespace) -> torch.utils.data.Dataset:
    # Get datatset
    DATASET = LIST_DATASETS.get(args.dataset.lower())
    if DATASET is None:
        raise ValueError(f"Dataset {args.dataset.lower()} not available.")

    # Shared parameters
    kwargs = {
        "rootdir": args.path_dataset,
        "input_feat": config["embedding"]["input_feat"],
        "voxel_size": config["embedding"]["voxel_size"],
        "num_neighbors": config["embedding"]["neighbors"],
        "dim_proj": config["waffleiron"]["dim_proj"],
        "grids_shape": config["waffleiron"]["grids_size"],
        "fov_xyz": config["waffleiron"]["fov_xyz"],
        "verbose": args.verbose,
    }

    # Get phase
    if args.eval:
        phase = "val"
    elif args.test:
        phase = "test"
    else:
        phase = "train"

    dataset = DATASET(
        phase=phase,
        **kwargs,
    )

    return dataset


def get_dataloader(
    dataset: torch.utils.data.Dataset, args: argparse.Namespace
) -> torch.utils.data.DataLoader:
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=None,
        drop_last=not args.eval and not args.test,
        collate_fn=Collate(),
    )

    return dataloader
