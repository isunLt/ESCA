import torch
import numpy as np
import pytorch_lightning as pl
# import lightning.pytorch as pl
from torch.utils.data import DataLoader
from utils.transforms import make_transforms_clouds
from downstream.dataloader_kitti import SemanticKITTIDataset
from downstream.dataloader_nuscenes import NuScenesDataset, custom_collate_fn
try:
    from downstream.dataloader_nuscenes_spvcnn import NuScenesDatasetSPVCNN, spvcnn_custom_collate_fn
except ImportError:
    NuScenesDatasetSPVCNN = None
    spvcnn_custom_collate_fn = None
from downstream.dataloader_waymo import WaymoDataset, custom_collate_fn_waymo
from downstream.dataloader_kitti import SemanticKITTIDataset, custom_collate_fn_semkitti
from downstream.dataloader_synth4d import SynthDataset, custom_collate_fn_synth4d
from downstream.dataloader_synth4d_spvcnn import Synth4DDatasetSPVCNN, spvcnn_custom_collate_fn_synth4d
from downstream.dataloader_kitti_spvcnn import SemanticKITTIDatasetSPVCNN, spvcnn_custom_collate_fn_semkitti
from downstream.dataloader_waymo_spvcnn import WaymoDatasetSPVCNN, spvcnn_custom_collate_fn_waymo
from downstream.dataloader_semanticposs import SemanticPOSSDataset, custom_collate_fn_semposs
from downstream.dataloader_semanticstf import SemanticSTFDataset, custom_collate_fn_semstf
from downstream.dataloader_rellis3D import Rellis3DDataset, custom_collate_fn_rellis3d
from downstream.dataloader_synlidar import SynLiDARDataset, custom_collate_fn_synlidar

class DownstreamDataModule(pl.LightningDataModule):
    """
    The equivalent of a DataLoader for pytorch lightning.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        # in multi-GPU the actual batch size is that
        self.batch_size = config["batch_size"] // config["num_gpus"]
        # the CPU workers are split across GPU
        self.num_workers = max(config["num_threads"] // config["num_gpus"], 1)

        self.model_type = self.config["model_points"]

    def setup(self, stage):
        # setup the dataloader: this function is automatically called by lightning
        transforms = make_transforms_clouds(self.config)
        if self.config["dataset"].lower() == "nuscenes" and "minkunet" in self.config["model_points"]:
            Dataset = NuScenesDataset
        elif self.config['dataset'].lower() == 'nuscenes' and self.config["model_points"] == "spvcnn":
            Dataset = NuScenesDatasetSPVCNN
        elif self.config["dataset"].lower() in ("kitti", "semantickitti") and "minkunet" in self.config["model_points"]:
            Dataset = SemanticKITTIDataset
        elif self.config["dataset"].lower() in ("kitti", "semantickitti") and self.config["model_points"] == "spvcnn":
            Dataset = SemanticKITTIDatasetSPVCNN
        elif self.config['dataset'].lower() == 'waymo' and "minkunet" in self.config["model_points"]:
            Dataset = WaymoDataset
        elif self.config['dataset'].lower() == 'waymo' and self.config["model_points"] == "spvcnn":
            Dataset = WaymoDatasetSPVCNN
        elif self.config['dataset'].lower() == 'synth4d' and self.config['model_points'] == 'minkunet':
            Dataset = SynthDataset
        elif self.config['dataset'].lower() == 'synth4d' and self.config['model_points'] == 'spvcnn':
            Dataset = Synth4DDatasetSPVCNN
        elif self.config['dataset'].lower() == 'semposs' and self.config['model_points'] == 'minkunet':
            Dataset = SemanticPOSSDataset
        elif self.config['dataset'].lower() == 'semstf' and self.config['model_points'] == 'minkunet':
            Dataset = SemanticSTFDataset
        elif self.config['dataset'].lower() == 'rellis3d' and self.config['model_points'] == 'minkunet':
            Dataset = Rellis3DDataset
        elif self.config['dataset'].lower() == 'synlidar' and self.config['model_points'] == 'minkunet':
            Dataset = SynLiDARDataset
        else:
            raise Exception(f"Unknown dataset {self.config['dataset']}")
        if self.config["training"] in ("parametrize", "parametrizing"):
            phase_train = "parametrizing"
            phase_val = "verifying"
        else:
            phase_train = "train"
            phase_val = "val"
        self.train_dataset = Dataset(
            phase=phase_train, transforms=transforms, config=self.config
        )
        if Dataset == NuScenesDataset:
            self.val_dataset = Dataset(
                phase=phase_val,
                config=self.config,
                cached_nuscenes=self.train_dataset.nusc,
            )
        else:
            self.val_dataset = Dataset(phase=phase_val, config=self.config)

    def train_dataloader(self):
        # construct the training dataloader: this function is automatically called
        # by lightning
        if self.model_type == 'spvcnn':
            collate_fn = spvcnn_custom_collate_fn
        elif self.config['dataset'].lower() in ("kitti", "semantickitti"):
            collate_fn = custom_collate_fn_semkitti
        elif self.config['dataset'].lower() == 'waymo':
            collate_fn = custom_collate_fn_waymo
        elif self.config['dataset'].lower() == 'synth4d':
            collate_fn = custom_collate_fn_synth4d
        elif self.config['dataset'].lower() == 'semposs':
            collate_fn = custom_collate_fn_semposs
        elif self.config['dataset'].lower() == 'semstf':
            collate_fn = custom_collate_fn_semstf
        elif self.config['dataset'].lower() == 'rellis3d':
            collate_fn = custom_collate_fn_rellis3d
        elif self.config['dataset'].lower() == 'synlidar':
            collate_fn = custom_collate_fn_synlidar
        else:
            collate_fn = custom_collate_fn
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=lambda id: np.random.seed(
                torch.initial_seed() // 2 ** 32 + id
            ),
        )

    def val_dataloader(self):
        # construct the validation dataloader: this function is automatically called
        # by lightning
        if self.model_type == 'spvcnn':
            collate_fn = spvcnn_custom_collate_fn
        elif self.config['dataset'].lower() in ("kitti", "semantickitti"):
            collate_fn = custom_collate_fn_semkitti
        elif self.config['dataset'].lower() == 'waymo':
            collate_fn = custom_collate_fn_waymo
        elif self.config['dataset'].lower() == 'synth4d':
            collate_fn = custom_collate_fn_synth4d
        elif self.config['dataset'].lower() == 'semposs':
            collate_fn = custom_collate_fn_semposs
        elif self.config['dataset'].lower() == 'semstf':
            collate_fn = custom_collate_fn_semstf
        elif self.config['dataset'].lower() == 'rellis3d':
            collate_fn = custom_collate_fn_rellis3d
        elif self.config['dataset'].lower() == 'synlidar':
            collate_fn = custom_collate_fn_synlidar
        else:
            collate_fn = custom_collate_fn
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=False,
            worker_init_fn=lambda id: np.random.seed(
                torch.initial_seed() // 2 ** 32 + id
            ),
        )
