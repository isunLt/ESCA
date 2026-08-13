import os
import argparse
import torch
import torch.nn as nn
import MinkowskiEngine as ME
import pytorch_lightning as pl
from utils.read_config import generate_config
from pretrain.model_builder import make_model
from pytorch_lightning.plugins import DDPPlugin
from pretrain.lightning_trainer import LightningPretrain
from pretrain.lightning_datamodule import PretrainDataModule
from pretrain.lightning_trainer_spconv import LightningPretrainSpconv


def main():
    """
    Code for launching the pretraining
    """
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument(
        "--cfg_file", type=str, default="config/slidr_minkunet.yaml", help="specify the config for training"
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

    if os.environ.get("LOCAL_RANK", 0) == 0:
        print(
            "\n" + "\n".join(list(map(lambda x: f"{x[0]:20}: {x[1]}", config.items())))
        )
    config["pretraining_path"] = args.pretraining_path

    if config['debug']:
        pl.seed_everything(config["seed"])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # print('torch.random.initial_seed():', torch.random.initial_seed())
    # print('torch.backends.cudnn.deterministic:', torch.backends.cudnn.deterministic)
    # print('torch.backends.cudnn.benchmark:', torch.backends.cudnn.benchmark)

    dm = PretrainDataModule(config)
    model_points, model_images = make_model(config, config["pretraining_path"])
    if config["num_gpus"] > 1:
        model_points = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model_points)
        model_images = nn.SyncBatchNorm.convert_sync_batchnorm(model_images)
    if  "minkunet" in config["model_points"]:
        print('====================using %s====================' % str(config["model_points"]))
        module = LightningPretrain(model_points, model_images, config)
    elif config["model_points"] == "voxelnet":
        module = LightningPretrainSpconv(model_points, model_images, config)
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
        # replace_sampler_ddp=False
    )
    print("Starting the training")
    trainer.fit(module, dm)


if __name__ == "__main__":
    main()

# v2
# import os
# import argparse
# import torch.nn as nn
# import torch
# torch.set_float32_matmul_precision('high')
# import MinkowskiEngine as ME
# # import pytorch_lightning as pl
# from utils.read_config import generate_config
# from pretrain.model_builder import make_model
# # from pytorch_lightning.plugins import DDPPlugin
# import lightning.pytorch as pl  # v2
# from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
# from pretrain.lightning_trainer import LightningPretrain
# from pretrain.lightning_datamodule import PretrainDataModule
# from pretrain.lightning_trainer_spconv import LightningPretrainSpconv
#
#
# def main():
#     """
#     Code for launching the pretraining
#     """
#     parser = argparse.ArgumentParser(description="arg parser")
#     parser.add_argument(
#         "--cfg_file", type=str, default="config/slidr_minkunet.yaml", help="specify the config for training"
#     )
#     parser.add_argument(
#         "--resume_path", type=str, default=None, help="provide a path to resume an incomplete training"
#     )
#     parser.add_argument(
#         '--run_dir', type=str, default='default_run', help='path to save the logs'
#     )
#     args = parser.parse_args()
#     config = generate_config(args.cfg_file)
#     config['run_dir'] = args.run_dir
#     if args.resume_path:
#         config['resume_path'] = args.resume_path
#
#     if os.environ.get("LOCAL_RANK", 0) == 0:
#         print(
#             "\n" + "\n".join(list(map(lambda x: f"{x[0]:20}: {x[1]}", config.items())))
#         )
#
#     dm = PretrainDataModule(config)
#     model_points, model_images = make_model(config)
#     if config["num_gpus"] > 1:
#         model_points = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model_points)
#         model_images = nn.SyncBatchNorm.convert_sync_batchnorm(model_images)
#     if config["model_points"] == "minkunet":
#         module = LightningPretrain(model_points, model_images, config)
#     elif config["model_points"] == "voxelnet":
#         module = LightningPretrainSpconv(model_points, model_images, config)
#     path = os.path.join(config["working_dir"], str(config["datetime"]) + '-' + str(config['run_dir']))
#     checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=os.path.join(path, 'checkpoints'))
#     tb_logger = TensorBoardLogger(save_dir=os.path.join(path, "tb_logs"))
#     csv_logger = CSVLogger(save_dir=os.path.join(path, "csv_logs"))
#     # trainer = pl.Trainer(
#     #     gpus=config["num_gpus"],
#     #     accelerator="ddp",
#     #     default_root_dir=path,
#     #     checkpoint_callback=True,
#     #     max_epochs=config["num_epochs"],
#     #     plugins=DDPPlugin(find_unused_parameters=False),
#     #     num_sanity_val_steps=0,
#     #     resume_from_checkpoint=config["resume_path"],
#     #     check_val_every_n_epoch=1,
#     #     # replace_sampler_ddp=False
#     # )
#     trainer = pl.Trainer(
#         # common
#         default_root_dir=path,
#         enable_checkpointing=True,
#         max_epochs=config["num_epochs"],
#         num_sanity_val_steps=0,
#         check_val_every_n_epoch=1,
#         # v2.0,
#         devices=config["num_gpus"],
#         accelerator="gpu",
#         callbacks=[checkpoint_callback],
#         logger=[tb_logger, csv_logger],
#         log_every_n_steps=50,
#         # strategy='ddp_find_unused_parameters_true'
#         # v1.0
#         # gpus=config["num_gpus"],
#         # accelerator="ddp",
#         # checkpoint_callback=True,
#         # resume_from_checkpoint=config["resume_path"],
#         # plugins=DDPPlugin(find_unused_parameters=False),
#     )
#     print("Starting the training")
#     trainer.fit(module, dm, ckpt_path=config["resume_path"])
#
#
# if __name__ == "__main__":
#     main()
