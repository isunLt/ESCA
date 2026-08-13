import os
import gc
import argparse
import torch
# torch.set_float32_matmul_precision('high')
import MinkowskiEngine as ME
import pytorch_lightning as pl
import torch.nn as nn
from downstream.evaluate import evaluate
from utils.read_config import generate_config
from downstream.model_builder import make_model
from pytorch_lightning.plugins import DDPPlugin
from downstream.panoptic_segmentation.lightning_datamodule import DownstreamDataModule
from downstream.panoptic_segmentation.lightning_trainer import LightningDownstream
from downstream.panoptic_segmentation.lightning_trainer_spconv import LightningDownstreamSpconv
from downstream.panoptic_segmentation.lightning_trainer_cylinder3d import LightningDownstreamCylinder3D
from downstream.panoptic_segmentation.model_builder_panoptic import make_model
# from utils.common_utils import create_logger
from utils.read_config import generate_config
def main():
    """
    Code for launching the downstream training
    """
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument(
        "--cfg_file", type=str, default="config/semseg_nuscenes.yaml", help="specify the config for training"
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
    if args.resume_path:
        config['resume_path'] = args.resume_path
    if args.pretraining_path:
        config['pretraining_path'] = args.pretraining_path

    path = os.path.join(config["working_dir"], str(config["datetime"]) + '-' + str(config['run_dir']))
    # path = os.path.join(config["working_dir"], f"{config['run_dir']}_{config['datetime']}")
    config['working_path'] = path
    print(f"Local Rank: {os.environ.get('LOCAL_RANK')}, working_path: {config['working_path']}")
    os.makedirs("./tmpdir", exist_ok=True)

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
    model_points_name = config['model_points'] if 'model_points' in config else 'minkunet'
    model_points_name = model_points_name.lower()
    if config["num_gpus"] > 1:
        if model_points_name == 'minkunet':
            model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)
        elif model_points_name == 'voxelnet':
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        elif model_points_name == 'cylinder3d':
            model = None
        elif model_points_name == 'cylinder3d_separate':
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if model_points_name == 'minkunet':
        module = LightningDownstream(model, config)
    elif model_points_name == 'voxelnet':
        module = LightningDownstreamSpconv(model, config)
    elif model_points_name == 'cylinder3d':
        module = LightningDownstreamSpconv(model, config)
    elif model_points_name == 'cylinder3d_separate':
        module = LightningDownstreamCylinder3D(model, config)
    else:
        raise Exception("Unknown model name")

    if model_points_name == 'cylinder3d':
        if config['pretraining_path'] is not None and os.path.exists(config['pretraining_path']):
            module.load_pretraining_file(config['pretraining_path'])

    # path = os.path.join(config["working_dir"], config["datetime"])

    # tb_logger, csv_logger = TensorBoardLogger(save_dir=path), CSVLogger(save_dir=path)
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


if __name__ == "__main__":
    main()