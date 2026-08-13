import torch
import numpy as np
import pytorch_lightning as pl
# import lightning.pytorch as pl  # v2
from torch.utils.data import DataLoader
try:
    from pretrain.dataloader_nuscenes import (
        NuScenesMatchDataset,
        minkunet_collate_pair_fn,
    )
except ImportError:
    NuScenesMatchDataset = None
    minkunet_collate_pair_fn = None
try:
    from pretrain.dataloader_nuscenes_spconv import NuScenesMatchDatasetSpconv, spconv_collate_pair_fn
except ImportError:
    NuScenesMatchDatasetSpconv = None
    spconv_collate_pair_fn = None
try:
    from pretrain.dataloader_nuscenes_sweeps_temporal_cluster_spconv import NuScenesSweepsTemporalClusterMatchDatasetSpconv, spconv_collate_pair_fn_sweeps_temporal_cluster
except ImportError:
    NuScenesSweepsTemporalClusterMatchDatasetSpconv = None
    spconv_collate_pair_fn_sweeps_temporal_cluster = None
try:
    from pretrain.dataloader_nuscenes_spvcnn import NuScenesMatchDatasetSPVCNN, spvcnn_collate_pair_fn
except ImportError:
    NuScenesMatchDatasetSPVCNN = None
    spvcnn_collate_pair_fn = None
try:
    from pretrain.dataloader_nuscenes_sweeps import NuScenesSweepsMatchDataset, minkunet_collate_pair_fn_sweeps
except ImportError:
    NuScenesSweepsMatchDataset = None
    minkunet_collate_pair_fn_sweeps = None
try:
    from pretrain.dataloader_nuscenes_sweeps_spvcnn import NuScenesSweepsMatchDatasetSPVCNN, spvcnn_collate_pair_fn
except ImportError:
    NuScenesSweepsMatchDatasetSPVCNN = None
    spvcnn_collate_pair_fn = None
try:
    from pretrain.dataloader_nuscenes_with_label import NuScenesDatasetWithLabel, custom_collate_fn
except ImportError:
    NuScenesDatasetWithLabel = None
    custom_collate_fn = None
try:
    from pretrain.dataloader_nuscenes_sweeps_find import NuScenesSweepsFindMatchDataset, minkunet_collate_pair_fn_sweeps_find
except ImportError:
    NuScenesSweepsFindMatchDataset = None
    minkunet_collate_pair_fn_sweeps_find = None
try:
    from pretrain.dataloader_nuscenes_sweeps_temporal import NuScenesSweepsTemporalMatchDataset, minkunet_collate_pair_fn_sweeps_temporal
except ImportError:
    NuScenesSweepsFindMatchDataset = None
    minkunet_collate_pair_fn_sweeps_temporal = None
try:
    from pretrain.dataloader_nuscenes_sweeps_temporal_cluster import NuScenesSweepsTemporalClusterMatchDataset, minkunet_collate_pair_fn_sweeps_temporal_cluster
except ImportError:
    NuScenesSweepsTemporalClusterMatchDataset = None
    minkunet_collate_pair_fn_sweeps_temporal_cluster = None
try:
    from pretrain.dataloader_nuscenes_sweeps_temporal_cluster_spvcnn import NuScenesSweepsTemporalClusterMatchDatasetSPVCNN, spvcnn_collate_pair_fn_sweeps_temporal_cluster
except ImportError:
    NuScenesSweepsTemporalClusterMatchDatasetSPVCNN = None
    spvcnn_collate_pair_fn_sweeps_temporal_cluster = None
try:
    from pretrain.dataloader_nuscenes_semkitti import NuScenesSemKITTIMatchDataset, minkunet_collate_pair_fn_nusc_semkitti
except ImportError:
    NuScenesSemKITTIMatchDataset = None
    minkunet_collate_pair_fn_nusc_semkitti = None
try:
    from pretrain.dataloader_concat import ConcatMatchDataset, minkunet_collate_pair_fn_concat
except ImportError:
    ConcatMatchDataset = None
    minkunet_collate_pair_fn_concat = None
try:
    from pretrain.dataloader_concat_spvcnn import ConcatMatchDatasetSPVCNN, spvcnn_collate_pair_fn_concat
except ImportError:
    ConcatMatchDatasetSPVCNN = None
    spvcnn_collate_pair_fn_concat = None
try:
    from pretrain.dataloader_nuscenes_cylinder3d import NuScenesDatasetCylinder3d, cylinder3d_custom_collate_fn
except ImportError:
    NuScenesDatasetCylinder3d = None
    cylinder3d_custom_collate_fn = None
from utils.transforms import (
    make_transforms_clouds,
    make_transforms_asymmetrical,
    make_transforms_asymmetrical_val,
)


class PretrainDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if config["num_gpus"]:
            self.batch_size = config["batch_size"] // config["num_gpus"]
        else:
            self.batch_size = config["batch_size"]

    def setup(self, stage):
        cloud_transforms_train = make_transforms_clouds(self.config)
        mixed_transforms_train = make_transforms_asymmetrical(self.config)
        cloud_transforms_val = None
        mixed_transforms_val = make_transforms_asymmetrical_val(self.config)
        if self.config["dataset"].lower() == "nuscenes" and "minkunet" in self.config["model_points"]:
            Dataset = NuScenesMatchDataset
        elif self.config["dataset"].lower() == "nuscenes" and self.config["model_points"] == "voxelnet":
            Dataset = NuScenesMatchDatasetSpconv
        elif self.config['dataset'].lower() == 'nuscenes' and self.config['model_points'] == 'spvcnn':
            Dataset = NuScenesMatchDatasetSPVCNN
        elif self.config['dataset'].lower() == 'nuscenes_sweeps' and 'minkunet' in self.config['model_points']:
            Dataset = NuScenesSweepsMatchDataset
        elif self.config['dataset'].lower() == 'nuscenes_sweeps' and self.config['model_points'] == 'spvcnn':
            Dataset = NuScenesSweepsMatchDatasetSPVCNN
        elif self.config['dataset'].lower() == 'nuscenes_orcale' and self.config['model_points'] == 'minkunet':
            print('--------------Warning! using nuscenes_orcale!-----------------')
            Dataset = NuScenesDatasetWithLabel
        elif self.config['dataset'].lower() == 'nuscenes_sweeps_find' and self.config['model_points'] == 'minkunet':
            Dataset = NuScenesSweepsFindMatchDataset
        elif self.config['dataset'].lower() == 'nuscenes_sweeps_temporal' and self.config['model_points'] == 'minkunet':
            Dataset = NuScenesSweepsTemporalMatchDataset
        elif self.config['dataset'].lower() == 'nuscenes_sweeps_temporal_cluster' and 'minkunet' in self.config['model_points']:
            Dataset = NuScenesSweepsTemporalClusterMatchDataset
        elif self.config['dataset'].lower() == 'nuscenes_sweeps_temporal_cluster' and self.config['model_points'] == 'voxelnet':
            Dataset = NuScenesSweepsTemporalClusterMatchDatasetSpconv
        elif self.config['dataset'].lower() == 'nuscenes_sweeps_temporal_cluster_spvcnn' and self.config['model_points'] == 'spvcnn':
            Dataset = NuScenesSweepsTemporalClusterMatchDatasetSPVCNN
        elif self.config['dataset'].lower() == 'nuscenes_semkitti' and 'minkunet' in self.config['model_points']:
            Dataset = NuScenesSemKITTIMatchDataset
        elif self.config['dataset'].lower() == 'concat_datasets' and 'minkunet' in self.config['model_points']:
            Dataset = ConcatMatchDataset
        elif self.config['dataset'].lower() == 'concat_datasets' and 'spvcnn' in self.config['model_points']:
            Dataset = ConcatMatchDatasetSPVCNN
        elif self.config['dataset'].lower() == 'nuscenes' and 'cylinder3d' in self.config['model_points']:
            Dataset = NuScenesDatasetCylinder3d
        else:
            raise Exception("Dataset Unknown")

        if self.config["training"] in ("parametrize", "parametrizing"):
            phase_train = "parametrizing"
            phase_val = "verifying"
        else:
            phase_train = "train"
            phase_val = "val"
        self.train_dataset = Dataset(
            phase=phase_train,
            shuffle=True,
            cloud_transforms=cloud_transforms_train,
            mixed_transforms=mixed_transforms_train,
            config=self.config,
        )
        if 'nuscenes_sweeps_temporal_cluster' in self.config['dataset'].lower():
            self.val_dataset = Dataset(
                phase=phase_val,
                shuffle=False,
                cloud_transforms=cloud_transforms_val,
                mixed_transforms=mixed_transforms_val,
                config=self.config,
                cached_nuscenes=self.train_dataset.nusc,
                cached_list_sweeps=self.train_dataset.list_sweeps
            )
        else:
            self.val_dataset = Dataset(
                phase=phase_val,
                shuffle=False,
                cloud_transforms=cloud_transforms_val,
                mixed_transforms=mixed_transforms_val,
                config=self.config,
                cached_nuscenes=self.train_dataset.nusc
            )

    def train_dataloader(self):

        if self.config["num_gpus"]:
            num_workers = self.config["num_threads"] // self.config["num_gpus"]
        else:
            num_workers = self.config["num_threads"]
        if "minkunet" in self.config["model_points"]:
            default_collate_pair_fn = minkunet_collate_pair_fn_sweeps
        elif self.config['model_points'] == 'spvcnn':
            default_collate_pair_fn = spvcnn_collate_pair_fn
        elif self.config['model_points'] == 'cylinder3d':
            default_collate_pair_fn = cylinder3d_custom_collate_fn
        else:
            default_collate_pair_fn = spconv_collate_pair_fn

        if self.config['dataset'] == 'nuscenes_orcale':
            default_collate_pair_fn = custom_collate_fn
        elif self.config['dataset'] == 'nuscenes_sweeps_find':
            default_collate_pair_fn = minkunet_collate_pair_fn_sweeps_find
        elif self.config['dataset'] == 'nuscenes_sweeps_temporal':
            default_collate_pair_fn = minkunet_collate_pair_fn_sweeps_temporal
        elif self.config['dataset'] == 'nuscenes_sweeps_temporal_cluster' and 'minkunet' in self.config['model_points']:
            default_collate_pair_fn = minkunet_collate_pair_fn_sweeps_temporal_cluster
        elif self.config['dataset'] == 'nuscenes_sweeps_temporal_cluster' and self.config['model_points'] == 'voxelnet':
            default_collate_pair_fn = spconv_collate_pair_fn_sweeps_temporal_cluster
        elif self.config['dataset'] == 'nuscenes_sweeps_temporal_cluster_spvcnn' and self.config['model_points'] == 'spvcnn':
            default_collate_pair_fn = spvcnn_collate_pair_fn_sweeps_temporal_cluster
        elif self.config['dataset'] == 'nuscenes_semkitti' and 'minkunet' in self.config['model_points']:
            default_collate_pair_fn = minkunet_collate_pair_fn_nusc_semkitti
        elif self.config['dataset'] == 'concat_datasets' and 'minkunet' in self.config['model_points']:
            default_collate_pair_fn = minkunet_collate_pair_fn_concat
        elif self.config['dataset'] == 'concat_datasets' and 'spvcnn' in self.config['model_points']:
            default_collate_pair_fn = spvcnn_collate_pair_fn_concat

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=default_collate_pair_fn,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=lambda id: np.random.seed(
                torch.initial_seed() // 2 ** 32 + id
            ),
        )

    def val_dataloader(self):

        if self.config["num_gpus"]:
            num_workers = self.config["num_threads"] // self.config["num_gpus"]
        else:
            num_workers = self.config["num_threads"]
        if "minkunet" in self.config["model_points"]:
            default_collate_pair_fn = minkunet_collate_pair_fn_sweeps
        elif self.config['model_points'] == 'spvcnn':
            default_collate_pair_fn = spvcnn_collate_pair_fn
        elif self.config['model_points'] == 'cylinder3d':
            default_collate_pair_fn = cylinder3d_custom_collate_fn
        else:
            default_collate_pair_fn = spconv_collate_pair_fn

        if self.config['dataset'] == 'nuscenes_orcale':
            default_collate_pair_fn = custom_collate_fn
        elif self.config['dataset'] == 'nuscenes_sweeps_find':
            default_collate_pair_fn = minkunet_collate_pair_fn_sweeps_find
        elif self.config['dataset'] == 'nuscenes_sweeps_temporal':
            default_collate_pair_fn = minkunet_collate_pair_fn_sweeps_temporal
        elif self.config['dataset'] == 'nuscenes_sweeps_temporal_cluster' and 'minkunet' in self.config['model_points']:
            default_collate_pair_fn = minkunet_collate_pair_fn_sweeps_temporal_cluster
        elif self.config['dataset'] == 'nuscenes_sweeps_temporal_cluster' and self.config['model_points'] == 'voxelnet':
            default_collate_pair_fn = spconv_collate_pair_fn_sweeps_temporal_cluster
        elif self.config['dataset'] == 'nuscenes_sweeps_temporal_cluster_spvcnn' and self.config['model_points'] == 'spvcnn':
            default_collate_pair_fn = spvcnn_collate_pair_fn_sweeps_temporal_cluster

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=default_collate_pair_fn,
            pin_memory=True,
            drop_last=False,
            worker_init_fn=lambda id: np.random.seed(
                torch.initial_seed() // 2 ** 32 + id
            ),
        )
