import os
import re
import torch
import numpy as np
from torch.utils.data import Dataset
from torchsparse import SparseTensor
from torchsparse.utils.quantize import sparse_quantize
from torchsparse.utils.collate import sparse_collate_fn, sparse_collate
from utils.transforms import make_transforms_clouds


TRAIN_SET = {0, 1, 2, 3, 4, 5, 6, 7, 9, 10}
VALIDATION_SET = {8}
TEST_SET = {11, 12, 13, 14, 15, 16, 17, 18, 19, 20}


def spvcnn_custom_collate_fn_semkitti(list_data):
    """
    Collate function adapted for creating batches with MinkowskiEngine.
    """
    input = list(zip(*list_data))
    labelized = len(input) == 5
    if labelized:
        xyz, voxel, labels, evaluation_labels, inverse_indexes = input
    else:
        xyz, voxel, inverse_indexes = input

    coords_batch, len_batch = [], []

    # for batch_id, coo in enumerate(coords):
    #     N = coords[batch_id].shape[0]
    #     coords_batch.append(
    #         torch.cat((torch.ones(N, 1, dtype=torch.int32) * batch_id, coo), 1)
    #     )
    #     len_batch.append(N)
    for batch_id in range(len(voxel)):
        N = voxel[batch_id].C.shape[0]
        len_batch.append(N)

    # Concatenate all lists
    # coords_batch = torch.cat(coords_batch, 0).int()
    # feats_batch = torch.cat(feats, 0).float()
    voxel_batch = sparse_collate(voxel)
    if labelized:
        # labels_batch = torch.cat(labels, 0).long()
        labels_batch = sparse_collate(labels)
        return {
            "pc": xyz,  # point cloud
            # "sinput_C": coords_batch,  # discrete coordinates (ME)
            # "sinput_F": feats_batch,  # point features (N, 3)
            'Input_P': voxel_batch,
            "len_batch": len_batch,  # length of each batch
            "labels": labels_batch,  # labels for each (voxelized) point
            "evaluation_labels": evaluation_labels,  # labels for each point
            "inverse_indexes": inverse_indexes,  # labels for each point
        }
    else:
        return {
            "pc": xyz,
            # "sinput_C": coords_batch,
            # "sinput_F": feats_batch,
            'Input_P': voxel_batch,
            "len_batch": len_batch,
            "inverse_indexes": inverse_indexes,
        }


class SemanticKITTIDatasetSPVCNN(Dataset):
    """
    Dataset returning a lidar scene and associated labels.
    Note that superpixels fonctionality have been removed.
    """

    def __init__(self, phase, config, transforms=None):
        self.phase = phase
        self.labels = self.phase != "test"
        self.transforms = transforms
        self.voxel_size = config["voxel_size"]
        self.cylinder = config["cylindrical_coordinates"]

        # a skip ratio can be used to reduce the dataset size
        # and accelerate experiments
        if phase == "train":
            try:
                skip_ratio = config["dataset_skip_step"]
            except KeyError:
                skip_ratio = 1
        else:
            skip_ratio = 1

        if phase in ("train", "parametrizing"):
            phase_set = TRAIN_SET
        elif phase in ("val", "verifying"):
            phase_set = VALIDATION_SET
        elif phase == "test":
            phase_set = TEST_SET

        self.list_files = []
        for num in phase_set:
            directory = next(
                os.walk(
                    f"datasets/semantickitti/sequences/{num:0>2d}/velodyne"
                )
            )
            self.list_files.extend(
                map(
                    lambda x: f"datasets/semantickitti/sequences/"
                    f"{num:0>2d}/velodyne/" + x,
                    directory[2],
                )
            )
        self.list_files = sorted(self.list_files)[::skip_ratio]

        # labels' names lookup table
        self.eval_labels = {
            0: 0, 1: 0, 10: 1, 11: 2, 13: 5, 15: 3, 16: 5, 18: 4, 20: 5, 30: 6, 31: 7,
            32: 8, 40: 9, 44: 10, 48: 11, 49: 12, 50: 13, 51: 14, 52: 0, 60: 9, 70: 15,
            71: 16, 72: 17, 80: 18, 81: 19, 99: 0, 252: 1, 253: 7, 254: 6, 255: 8,
            256: 5, 257: 5, 258: 4, 259: 5,
        }

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, idx):
        lidar_file = self.list_files[idx]
        points = np.fromfile(lidar_file, dtype=np.float32).reshape((-1, 4))
        # get the points (4th coordinate is the point intensity)
        pc = points[:, :3]
        if self.labels:
            lidarseg_labels_filename = re.sub(
                "bin", "label", re.sub("velodyne", "labels", lidar_file)
            )
            points_labels = (
                np.fromfile(lidarseg_labels_filename, dtype=np.uint32) & 0xFFFF
            )

        pc = torch.tensor(pc)

        # apply the transforms (augmentation)
        if self.transforms:
            pc = self.transforms(pc)

        if self.cylinder:
            # Transform to cylinder coordinate and scale for voxel size
            x, y, z = pc.T
            rho = torch.sqrt(x ** 2 + y ** 2) / self.voxel_size
            # corresponds to a split each 1°
            phi = torch.atan2(y, x) * 180 / np.pi
            z = z / self.voxel_size
            coords_aug = torch.cat((rho[:, None], phi[:, None], z[:, None]), 1)
        else:
            pc_n = pc.numpy()
            coords_aug = np.round(pc_n / self.voxel_size).astype(np.int32)
            coords_aug -= coords_aug.min(0, keepdims=1)

        # Voxelization
        _, indexes, inverse_indexes = sparse_quantize(
            coords_aug, return_index=True, return_inverse=True
        )
        # unique_feats = torch.tensor(points[indexes][:, 3:])
        unique_feats = torch.tensor(points[indexes])
        unique_feats[:, 3:] = unique_feats[:, 3:] + 1.
        discrete_coords = coords_aug[indexes]
        voxel = SparseTensor(unique_feats, discrete_coords)

        if self.labels:
            points_labels = torch.tensor(
                np.vectorize(self.eval_labels.__getitem__)(points_labels),
                dtype=torch.int32,
            )
            unique_labels = points_labels[indexes]
            unique_labels = SparseTensor(unique_labels, discrete_coords)

        if self.labels:
            return (
                pc,
                # discrete_coords,
                # unique_feats,
                voxel,
                unique_labels,
                points_labels,
                inverse_indexes,
            )
        else:
            return pc, voxel, inverse_indexes


def make_data_loader(config, phase, num_threads=0):
    """
    Create the data loader for a given phase and a number of threads.
    """
    # select the desired transformations
    if phase == "train":
        transforms = make_transforms_clouds(config)
    else:
        transforms = None

    # instantiate the dataset
    dset = SemanticKITTIDatasetSPVCNN(phase=phase, transforms=transforms, config=config)
    collate_fn = spvcnn_custom_collate_fn_semkitti
    batch_size = config["batch_size"] // config["num_gpus"]

    # create the loader
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=batch_size,
        # shuffle=False if sampler else True,
        shuffle=phase == "train",
        num_workers=num_threads,
        collate_fn=collate_fn,
        pin_memory=False,
        # sampler=sampler,
        drop_last=phase == "train",
        worker_init_fn=lambda id: np.random.seed(torch.initial_seed() // 2 ** 32 + id),
    )
    return loader
