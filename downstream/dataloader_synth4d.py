import os
import re
import torch
import numpy as np
from torch.utils.data import Dataset
from MinkowskiEngine.utils import sparse_quantize

import visualize_utils
from utils.transforms import make_transforms_clouds
import yaml
import pickle

def custom_collate_fn_synth4d(list_data):
    """
    Collate function adapted for creating batches with MinkowskiEngine.
    """
    input = list(zip(*list_data))
    labelized = len(input) == 6
    if labelized:
        xyz, coords, feats, labels, evaluation_labels, inverse_indexes = input
    else:
        xyz, coords, feats, inverse_indexes = input

    coords_batch, len_batch = [], []

    for batch_id, coo in enumerate(coords):
        N = coords[batch_id].shape[0]
        coords_batch.append(
            torch.cat((torch.ones(N, 1, dtype=torch.int32) * batch_id, coo), 1)
        )
        len_batch.append(N)

    # Concatenate all lists
    coords_batch = torch.cat(coords_batch, 0).int()
    feats_batch = torch.cat(feats, 0).float()
    if labelized:
        labels_batch = torch.cat(labels, 0).long()
        return {
            "pc": xyz,  # point cloud
            "sinput_C": coords_batch,  # discrete coordinates (ME)
            "sinput_F": feats_batch,  # point features (N, 3)
            "len_batch": len_batch,  # length of each batch
            "labels": labels_batch,  # labels for each (voxelized) point
            "evaluation_labels": evaluation_labels,  # labels for each point
            "inverse_indexes": inverse_indexes,  # labels for each point
        }
    else:
        return {
            "pc": xyz,
            "sinput_C": coords_batch,
            "sinput_F": feats_batch,
            "len_batch": len_batch,
            "inverse_indexes": inverse_indexes,
        }

class SynthDataset(Dataset):
    def __init__(self, phase, config, transforms=None):
        self.sensor = config['sensor']
        self.transforms = transforms
        self.voxel_size = config["voxel_size"]
        self.cylinder = config["cylindrical_coordinates"]

        if self.sensor == 'hdl64e':
            self.name = 'SyntheticKITTIDataset'
            self.dataset_path = os.path.join('datasets', 'kitti_synth')
        elif self.sensor == 'hdl32e':
            self.name = 'SyntheticNuScenesDataset'
            self.dataset_path = os.path.join('datasets', 'nuscenes_synth')
        else:
            raise NotImplementedError

        split = 'training_split' if phase == 'train' else 'validation_split'
        split_path = os.path.join(self.dataset_path, split)
        with open(split_path + '.pkl', 'rb') as f:
            self.split = pickle.load(f)

        remap_dict_val = {
            0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8,
            9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16,
            17: 17, 18: 18, 19: 19, 20: 20, 21: 21, 22: 22
        }
        max_key = max(remap_dict_val.keys())
        remap_lut_val = np.zeros((max_key + 100), dtype=np.int32)
        remap_lut_val[list(remap_dict_val.keys())] = list(remap_dict_val.values())

        self.remap_lut_val = remap_lut_val

        self.path_list = []

        for town in self.split.keys():
            pc_path = os.path.join(self.dataset_path, town, 'velodyne')
            self.path_list.extend([os.path.join(pc_path, str(f)+'.npy') for f in np.sort(self.split[town])])

        if phase == "train":
            try:
                skip_ratio = config["dataset_skip_step"]
            except KeyError:
                skip_ratio = 1
        else:
            skip_ratio = 1

        self.path_list = self.path_list[::skip_ratio]

    def __getitem__(self, i):
        pc_path = self.path_list[i]
        points = np.load(pc_path).astype(np.float32)

        dir, file = os.path.split(pc_path)
        label_path = os.path.join(dir, '../labels', file[:-4] + '.npy')

        if not os.path.exists(label_path):
            points_labels = np.zeros(np.shape(points)[0], dtype=np.int32)
        else:
            points_labels = np.load(label_path).astype(np.int32).reshape([-1])
            points_labels = self.remap_lut_val[points_labels]
            # points_labels = points_labels + 1  # make ignore label = 0
        points_labels = torch.tensor(points_labels, dtype=torch.int32)

        pc = points[:, :3]
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
            coords_aug = pc / self.voxel_size

        # Voxelization
        discrete_coords, indexes, inverse_indexes = sparse_quantize(
            coords_aug, return_index=True, return_inverse=True
        )
        unique_feats = torch.tensor(points[indexes][:, 3][..., np.newaxis])
        unique_labels = points_labels[indexes]
        # visualize_utils.visualize_pcd(discrete_coords, target=unique_labels)

        return (
            pc,
            discrete_coords,
            unique_feats,
            unique_labels,
            points_labels,
            inverse_indexes,
        )

        # if self.phase == 'train' and self.augment_data:
        #     sampled_idx = self.random_sample(points)
        #
        #     points = points[sampled_idx]
        #     colors = colors[sampled_idx]
        #     labels = labels[sampled_idx]
        #
        #     voxel_mtx, affine_mtx = self.voxelizer.get_transformation_matrix()
        #
        #     rigid_transformation = affine_mtx @ voxel_mtx
        #     # Apply transformations
        #
        #     homo_coords = np.hstack((points, np.ones((points.shape[0], 1), dtype=points.dtype)))
        #     # coords = np.floor(homo_coords @ rigid_transformation.T[:, :3])
        #     points = homo_coords @ rigid_transformation.T[:, :3]
        #
        # if self.ignore_label is None:
        #     vox_ign_label = -100
        # else:
        #     vox_ign_label = self.ignore_label
        #
        # quantized_coords, feats, labels = ME.utils.sparse_quantize(points,
        #                                                            colors,
        #                                                            labels=labels,
        #                                                            ignore_label=vox_ign_label,
        #                                                            quantization_size=self.voxel_size)
        #
        # if self.input_transforms is not None:
        #     quantized_coords, feats, labels = self.input_transforms(quantized_coords, feats, labels)
        #
        # return {"coordinates": quantized_coords,
        #         "features": torch.from_numpy(feats),
        #         "labels": torch.from_numpy(labels)}

    def __len__(self):
        return len(self.path_list)

    # def get_dataset_weights(self):
    #     weights = np.zeros(self.remap_lut_val.max()+1)
    #
    #     for l in tqdm.tqdm(range(len(self.path_list)), desc='Loading weights', leave=True):
    #         pc_path = self.path_list[l]
    #
    #         dir, file = os.path.split(pc_path)
    #         label_path = os.path.join(dir, '../labels', file[:-4] + '.npy')
    #
    #         if pc_path not in self.CACHE:
    #
    #             if os.path.exists(label_path):
    #                 labels = np.load(label_path).astype(np.int32).reshape([-1])
    #                 labels = self.remap_lut_val[labels]
    #                 lbl, count = np.unique(labels, return_counts=True)
    #                 if self.ignore_label is not None:
    #                     if self.ignore_label in lbl:
    #                         count = count[lbl != self.ignore_label]
    #                         lbl = lbl[lbl != self.ignore_label]
    #                 weights[lbl] += count
    #
    #     return weights

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
    dset = SynthDataset(phase=phase, transforms=transforms, config=config)
    collate_fn = custom_collate_fn_synth4d
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
