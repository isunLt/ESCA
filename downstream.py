import os
import gc
import argparse
import torch
# torch.set_float32_matmul_precision('high')
import MinkowskiEngine as ME
import pytorch_lightning as pl
# import lightning.pytorch as pl  # v2
# from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from downstream.evaluate import evaluate
from utils.read_config import generate_config
from downstream.model_builder import make_model
from pytorch_lightning.plugins import DDPPlugin
from downstream.lightning_trainer import LightningDownstream
from downstream.lightning_datamodule import DownstreamDataModule
from downstream.dataloader_kitti import make_data_loader as make_data_loader_kitti
from downstream.dataloader_nuscenes import make_data_loader as make_data_loader_nuscenes
from downstream.dataloader_waymo import make_data_loader as make_data_loader_waymo
from downstream.dataloader_synth4d import make_data_loader as make_data_loader_synth4d
from downstream.dataloader_semanticposs import make_data_loader as make_data_loader_semanticposs
from downstream.dataloader_semanticstf import make_data_loader as make_data_loader_semanticstf
from downstream.dataloader_rellis3D import make_data_loader as make_data_loader_rellis3d
from downstream.dataloader_synlidar import make_data_loader as make_data_loader_synlidar

def main():
    """
    Code for launching the downstream training
    """
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument(
        "--cfg_file", type=str, default="config/semseg_nuscenes.yaml", help="specify the config for training"
    )
    parser.add_argument(
        "--dataset_skip", type=int, default=None, help="set dataset_skip_step in config"
    )
    parser.add_argument(
        "--resume_path", type=str, default=None, help="provide a path to resume an incomplete training"
    )
    parser.add_argument(
        "--pretraining_path", type=str, default=None, help="provide a path to pre-trained weights"
    )
    parser.add_argument(
        '--run_dir', type=str, default='default_run', help='path to save the logs'
    )
    args = parser.parse_args()
    config = generate_config(args.cfg_file)
    config['run_dir'] = args.run_dir
    if args.dataset_skip is not None:
        config['dataset_skip_step'] = args.dataset_skip
        suffix_map = {
            100: '0p01', 20: '0p05', 10: '0p1', 4: '0p25', 1: '1p0'
        }
        config['run_dir'] = config['run_dir'] + '_finetune' + suffix_map[int(config['dataset_skip_step'])]
    if args.resume_path:
        config['resume_path'] = args.resume_path
    if args.pretraining_path:
        config['pretraining_path'] = args.pretraining_path

    if os.environ.get("LOCAL_RANK", 0) == 0:
        print(
            "\n" + "\n".join(list(map(lambda x: f"{x[0]:20}: {x[1]}", config.items())))
        )

    if config['debug']:
        pl.seed_everything(config["seed"])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    dm = DownstreamDataModule(config)
    model = make_model(config, config["pretraining_path"])
    if config["num_gpus"] > 1:
        model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)
    module = LightningDownstream(model, config)
    path = os.path.join(config["working_dir"], str(config["datetime"]) + '-' + str(config['run_dir']))
    trainer = pl.Trainer(
        gpus=config["num_gpus"],
        accelerator="ddp",
        default_root_dir=path,
        checkpoint_callback=True,
        max_epochs=config["num_epochs"],
        plugins=DDPPlugin(find_unused_parameters=False),
        num_sanity_val_steps=0,
        resume_from_checkpoint=config["resume_path"],
        check_val_every_n_epoch=1,
    )
    print("Starting the training")
    trainer.fit(module, dm)
    working_dir = module.working_dir
    print("Training finished, now evaluating the results")
    del trainer
    del dm
    del module
    gc.collect()
    if config["dataset"].lower() == "nuscenes":
        phase = "verifying" if config['training'] in ("parametrize", "parametrizing") else "val"
        val_dataloader = make_data_loader_nuscenes(
            config, phase, num_threads=config["num_threads"]
        )
    elif config["dataset"].lower() == "kitti":
        val_dataloader = make_data_loader_kitti(
            config, "val", num_threads=config["num_threads"]
        )
    elif config['dataset'].lower() == 'waymo':
        val_dataloader = make_data_loader_waymo(
            config, 'val', num_threads=config['num_threads']
        )
    elif config['dataset'].lower() == 'synth4d':
        val_dataloader = make_data_loader_synth4d(
            config, 'val', num_threads=config['num_threads']
        )
    elif config['dataset'].lower() == 'semposs':
        val_dataloader = make_data_loader_semanticposs(
            config, 'val', num_threads=config['num_threads']
        )
    elif config['dataset'].lower() == 'semstf':
        val_dataloader = make_data_loader_semanticstf(
            config, 'val', num_threads=config['num_threads']
        )
    elif config['dataset'].lower() == 'rellis3d':
        val_dataloader = make_data_loader_rellis3d(
            config, 'val', num_threads=config['num_threads']
        )
    elif config['dataset'].lower() == 'synlidar':
        val_dataloader = make_data_loader_synlidar(
            config, 'val', num_threads=config['num_threads']
        )
    path = os.path.join(working_dir, "best_miou_model.pt")
    checkpoint = torch.load(path, map_location='cpu')
    if "config" in checkpoint:
        for cfg in ("voxel_size", "cylindrical_coordinates"):
            assert checkpoint["config"][cfg] == config[cfg], (
                f"{cfg} is not consistant.\n"
                f"Checkpoint: {checkpoint['config'][cfg]}\n"
                f"Config: {config[cfg]}."
            )
    try:
        model.load_state_dict(checkpoint["model_points"])
        print('load checkpoint from', path)
    except KeyError:
        weights = {
            k.replace("model.", ""): v
            for k, v in checkpoint["state_dict"].items()
            if k.startswith("model.")
        }
        model.load_state_dict(weights)
        print('load checkpoint from', path)
    evaluate(model.to(0), val_dataloader, config)


if __name__ == "__main__":
    main()