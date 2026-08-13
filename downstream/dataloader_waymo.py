import os
import torch
import numpy as np
from torch.utils.data import Dataset
from MinkowskiEngine.utils import sparse_quantize
from utils.transforms import make_transforms_clouds


def custom_collate_fn_waymo(list_data):
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


class WaymoDataset(Dataset):

    def __init__(self, phase, config, transforms=None):
        self.phase = phase
        self.labels = self.phase != "test"
        self.transforms = transforms
        self.voxel_size = config["voxel_size"]
        self.cylinder = config["cylindrical_coordinates"]
        if config['using_coord_shift']:
            self.coord_shift = np.array(config['waymo_coord_shift'], dtype=np.float32).reshape(1, 3)
        else:
            self.coord_shift = None

        try:
            self.root = config['dataroot']
        except KeyError:
            self.root = 'datasets/waymo_bin_full'

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
            self.root = os.path.join(self.root, 'training')
        elif phase in ("val", "verifying"):
            self.root = os.path.join(self.root, 'validation')
        elif phase == "test":
            self.root = os.path.join(self.root, 'testing')

        self.seqs = sorted(os.listdir(self.root))
        self.pcd_files = []
        for s_i, seq in enumerate(self.seqs):
            if not os.path.isdir(os.path.join(self.root, str(seq))):
                continue
            seq_path = os.path.join(self.root, str(seq), 'lidar')
            seq_list = []
            for p_i, pcdname in enumerate(sorted(os.listdir(seq_path))):
                seq_list.append(os.path.join(seq_path, str(pcdname)))
            self.pcd_files.append(seq_list)

        self.id_pair = []
        self.keyframe_pair = []
        for s_i, seq in enumerate(self.pcd_files):
            for p_i, _ in enumerate(seq):
                self.id_pair.append((s_i, p_i))
                pcdfile = self.pcd_files[s_i][p_i]
                if os.path.exists(pcdfile.replace('lidar', 'label')):
                    self.keyframe_pair.append((s_i, p_i))

        self.keyframe_pair = sorted(self.keyframe_pair)[::skip_ratio]
        self.lidar_names = [1, 2, 3, 4, 5]

        # labels' names lookup table
        # self.eval_labels = {
        #     0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10,
        #     11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 50: 13, 51: 14, 52: 0, 60: 9, 70: 15,
        #     71: 16, 72: 17, 80: 18, 81: 19, 99: 0, 252: 1, 253: 7, 254: 6, 255: 8,
        #     256: 5, 257: 5, 258: 4, 259: 5,
        # }

    def __len__(self):
        return len(self.keyframe_pair)

    def _load_sensor_mask(self, s_i, p_i, ri):
        filepath = self.pcd_files[s_i][p_i]
        sensor_path = filepath.replace('lidar', 'sensor')
        if ri == 1:
            sensor_path = str(sensor_path).replace('sensor', 'sensor_ri2')
        return np.fromfile(sensor_path, dtype=np.uint8) == self.lidar_names[0]

    def _load_pcd(self, s_i, p_i, top_m, ri):
        filepath = self.pcd_files[s_i][p_i]
        labelpath = filepath.replace('lidar', 'label')
        if ri == 1:
            filepath = str(filepath).replace('lidar', 'lidar_ri2')
            labelpath = str(labelpath).replace('label', 'label_ri2')
        pts = np.fromfile(filepath, dtype=np.float32).reshape((-1, 6))
        xyz, i, r, e = pts[:, :3], np.tanh(pts[:, 3]), pts[:, 4], pts[:, 5]
        pts = np.concatenate([xyz, i.reshape([-1, 1]), e.reshape([-1, 1])], axis=-1)[:, :4]
        pts = pts[top_m]
        if self.phase == 'test':
            labels_ = np.expand_dims(np.zeros_like(pts[:, 0], dtype=int), axis=1)
        else:
            labels_ = np.fromfile(labelpath, dtype=np.int32).reshape([-1, 2])
            labels_ = labels_[top_m, 1]
        return pts, labels_

    def _load_single_frame(self, s_i, p_i):
        ri_list = [0, 1]
        filepath = self.pcd_files[s_i][p_i]
        pts_list, label_list = [], []
        for ri in ri_list:
            top_m = self._load_sensor_mask(s_i, p_i, ri)
            pts, labels_raw = self._load_pcd(s_i, p_i, top_m, ri)
            pts_list.append(pts)
            label_list.append(labels_raw)
        pts_list = np.concatenate(pts_list, axis=0)
        label_list = np.concatenate(label_list, axis=0)
        pose = np.fromfile(filepath.replace('lidar', 'pose'), dtype=np.float32).reshape([4, 4])
        return pts_list, label_list, pose

    def __getitem__(self, idx):
        s_i, p_i = self.keyframe_pair[idx]
        points, points_labels, pose = self._load_single_frame(s_i, p_i)

        if self.coord_shift is not None:
            points[:, :3] += self.coord_shift

        # get the points (4th coordinate is the point intensity)
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
        # unique_feats = torch.tensor(points[indexes][:, 3:] + 1.)
        unique_feats = torch.tensor(points[indexes][:, 3:])

        if self.labels:
            points_labels = torch.tensor(points_labels, dtype=torch.int32)
            unique_labels = points_labels[indexes]

        if self.labels:
            return (
                pc,
                discrete_coords,
                unique_feats,
                unique_labels,
                points_labels,
                inverse_indexes,
            )
        else:
            return pc, discrete_coords, unique_feats, inverse_indexes


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
    dset = WaymoDataset(phase=phase, transforms=transforms, config=config)
    collate_fn = custom_collate_fn_waymo
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