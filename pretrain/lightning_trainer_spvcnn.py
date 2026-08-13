import os
import re
import torch
import numpy as np
import torch.optim as optim
import pytorch_lightning as pl
from pretrain.criterion import NCELoss, SupConLoss, SupConLossIntra
from pytorch_lightning.utilities import rank_zero_only


class LightningPretrain(pl.LightningModule):
    def __init__(self, model_points, model_images, config):
        super().__init__()
        self.model_points = model_points
        self.model_images = model_images
        self._config = config
        self.losses = config["losses"]
        self.train_losses = []
        self.val_losses = []
        self.num_matches = config["num_matches"]
        self.batch_size = config["batch_size"]
        self.num_epochs = config["num_epochs"]
        self.superpixel_size = config["superpixel_size"]
        self.epoch = 0
        if config["resume_path"] is not None:
            self.epoch = int(
                re.search(r"(?<=epoch=)[0-9]+", config["resume_path"])[0]
            )
        self.nce = NCELoss(temperature=config["NCE_temperature"])
        self.supcon = SupConLoss(temperature=config["NCE_temperature"])
        self.supcon_intra = SupConLossIntra(temperature=config['NCE_temperature'])
        self.working_dir = os.path.join(config["working_dir"], str(config["datetime"]) + '-' + str(config['run_dir']))
        if os.environ.get("LOCAL_RANK", 0) == 0:
            os.makedirs(self.working_dir, exist_ok=True)
        self.ignored_label = config['ignored_label']

    def configure_optimizers(self):
        optimizer = optim.SGD(
            list(self.model_points.parameters()) + list(self.model_images.parameters()),
            lr=self._config["lr"],
            momentum=self._config["sgd_momentum"],
            dampening=self._config["sgd_dampening"],
            weight_decay=self._config["weight_decay"],
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.num_epochs)
        return [optimizer], [scheduler]

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)

    def training_step(self, batch, batch_idx):
        sparse_input = batch['Input_P'].cuda()
        output_points = self.model_points(sparse_input)
        self.model_images.eval()
        self.model_images.decoder.train()
        output_images = self.model_images(batch["input_I"])

        losses = [
            getattr(self, loss)(batch, output_points, output_images)
            for loss in self.losses
        ]
        loss = torch.mean(torch.stack(losses))

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size
        )
        self.train_losses.append(loss.detach().cpu())
        return loss

    def loss(self, batch, output_points, output_images):
        pairing_points = batch["pairing_points"]
        pairing_images = batch["pairing_images"]
        idx = np.random.choice(pairing_points.shape[0], self.num_matches, replace=False)
        k = output_points[pairing_points[idx]]
        m = tuple(pairing_images[idx].T.long())
        q = output_images.permute(0, 2, 3, 1)[m]
        return self.criterion(k, q)

    def loss_superpixels_average(self, batch, output_points, output_images):
        # compute a superpoints to superpixels loss using superpixels
        # torch.cuda.empty_cache()  # This method is extremely memory intensive
        superpixels = batch["superpixels"].long()  # [batch_size*6, H, W,]
        pairing_images = batch["pairing_images"]  # 每个点对应的像素, [N_fov, 3], (图片idx, x, y)
        pairing_points = batch["pairing_points"]  # 有对应像素的点的idx, [N_fov,]

        superpixels = (
            torch.arange(
                0,
                output_images.shape[0] * self.superpixel_size,
                self.superpixel_size,
                device=self.device,
            )[:, None, None] + superpixels
        )

        m = tuple(pairing_images.cpu().T.long())

        superpixels_I = superpixels.flatten()
        idx_P = torch.arange(pairing_points.shape[0], device=superpixels.device)
        total_pixels = superpixels_I.shape[0]
        idx_I = torch.arange(total_pixels, device=superpixels.device)

        with torch.no_grad():
            one_hot_P = torch.sparse_coo_tensor(
                torch.stack((
                    superpixels[m], idx_P  # 超像素id, 点id的id
                ), dim=0),
                torch.ones(pairing_points.shape[0], device=superpixels.device),
                (self.superpixel_size * superpixels.size(0), pairing_points.shape[0])
            )

            one_hot_I = torch.sparse_coo_tensor(
                torch.stack((
                    superpixels_I, idx_I
                ), dim=0),
                torch.ones(total_pixels, device=superpixels.device),
                (self.superpixel_size * superpixels.size(0), total_pixels)
            )

        k = one_hot_P @ output_points[pairing_points]
        k = k / (torch.sparse.sum(one_hot_P, 1).to_dense()[:, None] + 1e-6)
        q = one_hot_I @ output_images.permute(0, 2, 3, 1).flatten(0, 2)
        q = q / (torch.sparse.sum(one_hot_I, 1).to_dense()[:, None] + 1e-6)

        sp_l = torch.arange(0, k.size(0), device=self.device) % self.superpixel_size
        mask = torch.where(torch.logical_and(k[:, 0] != 0, sp_l != self.ignored_label))
        k = k[mask]
        q = q[mask]
        sp_l = sp_l[mask]

        # return self.supcon(k, q, sp_l) + self.supcon_intra(k, k, sp_l)
        return self.supcon(k, q, sp_l)
        # return self.nce(k, q) + self.supcon_intra(k, k, sp_l)

    # def loss_superpixels_average(self, batch, output_points, output_images):
    #     # compute a superpoints to superpixels loss using superpixels
    #     # torch.cuda.empty_cache()  # This method is extremely memory intensive
    #     superpixels = batch["superpixels"].long()  # [batch_size*6, H, W,]
    #     pairing_images = batch["pairing_images"]  # 每个点对应的像素, [N_fov, 3], (图片idx, x, y)
    #     pairing_points = batch["pairing_points"]  # 有对应像素的点的idx, [N_fov,]
    #
    #     superpixels = (
    #         torch.arange(
    #             0,
    #             output_images.shape[0] * self.superpixel_size,
    #             self.superpixel_size,
    #             device=self.device,
    #         )[:, None, None] + superpixels
    #     )
    #
    #     m = tuple(pairing_images.cpu().T.long())
    #
    #     superpixels_I = superpixels.flatten()
    #
    #     idx_P = torch.arange(pairing_points.shape[0], device=superpixels.device)
    #     total_pixels = superpixels_I.shape[0]
    #     idx_I = torch.arange(total_pixels, device=superpixels.device)
    #
    #     with torch.no_grad():
    #         one_hot_P = torch.sparse_coo_tensor(
    #             torch.stack((
    #                 superpixels[m], idx_P  # 超像素id, 点id的id
    #             ), dim=0),
    #             torch.ones(pairing_points.shape[0], device=superpixels.device),
    #             (superpixels.shape[0] * self.superpixel_size, pairing_points.shape[0])
    #         )
    #
    #         one_hot_I = torch.sparse_coo_tensor(
    #             torch.stack((
    #                 superpixels_I, idx_I
    #             ), dim=0),
    #             torch.ones(total_pixels, device=superpixels.device),
    #             (superpixels.shape[0] * self.superpixel_size, total_pixels)
    #         )
    #
    #     k = one_hot_P @ output_points[pairing_points]
    #     k = k / (torch.sparse.sum(one_hot_P, 1).to_dense()[:, None] + 1e-6)
    #     q = one_hot_I @ output_images.permute(0, 2, 3, 1).flatten(0, 2)
    #     q = q / (torch.sparse.sum(one_hot_I, 1).to_dense()[:, None] + 1e-6)
    #
    #     k = k[1:, :]
    #     q = q[1:, :]
    #     mask = torch.where(k[:, 0] != 0)
    #     k = k[mask]
    #     q = q[mask]
    #
    #     return self.criterion(k, q)

    def training_epoch_end(self, outputs):
        self.epoch += 1
        if self.epoch == self.num_epochs:
            self.save()
        return super().training_epoch_end(outputs)

    def validation_step(self, batch, batch_idx):
        sparse_input = batch['Input_P'].cuda()
        output_points = self.model_points(sparse_input)
        self.model_images.eval()
        output_images = self.model_images(batch["input_I"])

        losses = [
            getattr(self, loss)(batch, output_points, output_images)
            for loss in self.losses
        ]
        loss = torch.mean(torch.stack(losses))
        self.val_losses.append(loss.detach().cpu())

        self.log(
            "val_loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=self.batch_size
        )
        return loss

    @rank_zero_only
    def save(self):
        path = os.path.join(self.working_dir, "model.pt")
        torch.save(
            {
                "model_points": self.model_points.state_dict(),
                "model_images": self.model_images.state_dict(),
                "epoch": self.epoch,
                "config": self._config,
            },
            path,
        )
