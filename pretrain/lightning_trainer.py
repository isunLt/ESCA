import os
import re
import torch
import numpy as np
import torch.optim as optim
import MinkowskiEngine as ME
import pytorch_lightning as pl
from pretrain.criterion import NCELoss, SupConLoss, SupConLossIntra, SupConLossTemporal
from pytorch_lightning.utilities import rank_zero_only
import torch_scatter
import visualize_utils


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
        # self.criterion = NCELoss(temperature=config["NCE_temperature"])
        self.supcon = SupConLoss(temperature=config["NCE_temperature"])
        self.supcon_intra = SupConLossIntra(temperature=config['NCE_temperature'])
        self.nce = NCELoss(temperature=config["NCE_temperature"])
        # self.nce_clus = SupConLossTemporal(temperature=config['NCE_temperature'])
        self.working_dir = os.path.join(config["working_dir"], str(config["datetime"]) + '-' + str(config['run_dir']))
        if os.environ.get("LOCAL_RANK", 0) == 0:
            os.makedirs(self.working_dir, exist_ok=True)
        self.ignored_label = config['ignored_label']
        self.decoupled_head = config['decoupled_head']
        self.src_datasets = config.get('src_datasets', None)

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

    def _vis(self, batch):
        img_list = batch['input_I'].permute(0, 2, 3, 1).contiguous() * 255
        sp_list = batch['superpixels']
        for b_i in range(img_list.size(0)):
            visualize_utils.visualize_img(img_list[b_i], superpixel=sp_list[b_i])

    def _vis_pcd(self, batch):

        def get_sparse_image_pred(batch):
            pairing_images = batch["pairing_images"]
            logits_pairing_images = batch['superpixels'][pairing_images[:, 0], pairing_images[:, 1], pairing_images[:, 2]]
            return logits_pairing_images

        color = get_sparse_image_pred(batch)
        point = batch['sinput_C'][batch['pairing_points']]

        for b_i in range(8):
            m = point[:, 0] == b_i
            visualize_utils.visualize_pcd(point[m, 1:], predict=color[m])

    # def training_step(self, batch, batch_idx):
    #     sparse_input = ME.SparseTensor(batch["sinput_F"], batch["sinput_C"])
    #     output_points = self.model_points(sparse_input).F
    #     self.model_images.eval()
    #     if hasattr(self.model_images.encoder, 'set_train'):
    #         self.model_images.encoder.set_train()
    #     # self.model_images.encoder.layer4[2].train()
    #     # self._vis_pcd(batch)
    #     # self._vis(batch)
    #     self.model_images.decoder.train()
    #     output_images = self.model_images(batch["input_I"])
    #     # each loss is applied independtly on each GPU
    #     losses = [
    #         getattr(self, loss)(batch, output_points, output_images)
    #         for loss in self.losses
    #     ]
    #     loss = torch.mean(torch.stack(losses))
    #     # print('after_training_step:', batch['lidar_tokens'])
    #     # p2c_loss =  self.loss_p2c_maxpool(batch, output_points)
    #     # loss += p2c_loss
    #     # torch.cuda.empty_cache()
    #     self.log(
    #         "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size
    #     )
    #     # self.log(
    #     #     "p2c_loss", p2c_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size
    #     # )
    #     self.train_losses.append(loss.detach().cpu())
    #     return loss

    def training_step(self, batch, batch_idx):
        if self.decoupled_head:
            sparse_input = ME.SparseTensor(batch["sinput_F"], batch["sinput_C"])
            output_points, output_points_tmp = self.model_points(sparse_input)
            self.model_images.eval()
            if hasattr(self.model_images.encoder, 'set_train'):
                self.model_images.encoder.set_train()
            self.model_images.decoder.train()
            output_images, output_images_tmp = self.model_images(batch["input_I"])
            # each loss is applied independtly on each GPU
            # losses = [
            #     self.loss_superpixels_average(batch, output_points, output_images),
            #     self.loss_nce(batch, output_points_tmp, output_images_tmp),
            # ]
            # loss = torch.mean(torch.stack(losses))
            nce_loss, nce_loss_intra = self.loss_nce(batch, output_points_tmp, output_images_tmp)
            # nce_loss = self.loss_nce_with_conditions(batch, output_points_tmp, output_images_tmp)
            supcon_loss, supcon_intra_loss = self.loss_superpixels_average(batch, output_points, output_images)
            # loss = (nce_loss + supcon_loss) / 2 + (nce_loss_intra + supcon_intra_loss) / 2
            loss = nce_loss + supcon_loss + nce_loss_intra + supcon_intra_loss
            # loss = (nce_loss + supcon_loss + supcon_intra_loss) / 3
            self.log('nce_loss', nce_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
            self.log('nce_loss_intra', nce_loss_intra, on_step=True, on_epoch=True, prog_bar=True, logger=True,
                     batch_size=self.batch_size)
            self.log(
                "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size
            )
            self.train_losses.append(loss.detach().cpu())
            return loss
        else:
            # self._vis(batch)
            # print('trainer')
            # self._vis_pcd(batch)
            sparse_input = ME.SparseTensor(batch["sinput_F"], batch["sinput_C"])
            output_points = self.model_points(sparse_input).F
            self.model_images.eval()
            if hasattr(self.model_images.encoder, 'set_train'):
                self.model_images.encoder.set_train()
            self.model_images.decoder.train()
            output_images = self.model_images(batch["input_I"])
            # each loss is applied independtly on each GPU
            losses = [
                getattr(self, loss)(batch, output_points, output_images)
                for loss in self.losses
            ]
            loss = torch.mean(torch.stack(losses))
            # nce_loss = self.loss_nce(batch, output_points, output_images)
            # supcon_loss, supcon_intra_loss = self.loss_superpixels_average(batch, output_points, output_images)
            # loss = nce_loss + supcon_loss + supcon_intra_loss
            # loss = (nce_loss + supcon_loss + supcon_intra_loss) / 3
            # self.log('nce_loss', nce_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,
            #          batch_size=self.batch_size)
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

    def loss_p2c_maxpool(self, batch, output_points):
        superpixels = batch["superpixels"].long()  # [batch_size*6, H, W,]
        pairing_images = batch["pairing_images"]  # 每个点对应的像素, [N_fov, 3], (图片idx, x, y)
        pairing_points = batch["pairing_points"]  # 有对应像素的点的idx, [N_fov,]

        superpixels = (
            torch.arange(
                0,
                superpixels.shape[0] * self.superpixel_size,
                self.superpixel_size,
                device=self.device,
            )[:, None, None] + superpixels
        )

        pairing_superpixel = superpixels[pairing_images[:, 0], pairing_images[:, 1], pairing_images[:, 2]]
        pairing_points_feat = output_points[pairing_points, :]

        clus_feat = torch_scatter.scatter_max(pairing_points_feat, pairing_superpixel, dim=0)[0]
        clus_feat = torch.gather(clus_feat, dim=0, index=pairing_superpixel.view(-1, 1).expand(-1, output_points.size(-1)))
        loss = torch.nn.functional.mse_loss(pairing_points_feat, clus_feat, reduction='none').sum(-1).mean()
        return loss

    def loss_nce(self, batch, output_points, output_images):
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
                (superpixels.shape[0] * self.superpixel_size, pairing_points.shape[0])
            )

            one_hot_I = torch.sparse_coo_tensor(
                torch.stack((
                    superpixels_I, idx_I
                ), dim=0),
                torch.ones(total_pixels, device=superpixels.device),
                (superpixels.shape[0] * self.superpixel_size, total_pixels)
            )

        k = one_hot_P @ output_points[pairing_points]
        k = k / (torch.sparse.sum(one_hot_P, 1).to_dense()[:, None] + 1e-6)
        q = one_hot_I @ output_images.permute(0, 2, 3, 1).flatten(0, 2)
        q = q / (torch.sparse.sum(one_hot_I, 1).to_dense()[:, None] + 1e-6)

        mask = torch.where(k[:, 0] != 0)
        k = k[mask]
        q = q[mask]

        return self.nce(k, q)

        # return self.nce(k, q), self.nce(k, k)

    def loss_nce_with_conditions(self, batch, output_points, output_images):
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
                (superpixels.shape[0] * self.superpixel_size, pairing_points.shape[0])
            )

            one_hot_I = torch.sparse_coo_tensor(
                torch.stack((
                    superpixels_I, idx_I
                ), dim=0),
                torch.ones(total_pixels, device=superpixels.device),
                (superpixels.shape[0] * self.superpixel_size, total_pixels)
            )

        k = one_hot_P @ output_points[pairing_points]
        k = k / (torch.sparse.sum(one_hot_P, 1).to_dense()[:, None] + 1e-6)
        q = one_hot_I @ output_images.permute(0, 2, 3, 1).flatten(0, 2)
        q = q / (torch.sparse.sum(one_hot_I, 1).to_dense()[:, None] + 1e-6)

        # nce = 0.
        nce = []

        dset_mask = batch['sinput_C'][:, 0][pairing_points]
        conditions = torch.tensor([self.src_datasets.index(c) for c in batch['conditions']], dtype=torch.uint8,
                                  device=dset_mask.device)
        dset_mask = conditions[dset_mask]
        p2sp = superpixels[pairing_images[:, 0], pairing_images[:, 1], pairing_images[:, 2]]
        for dset_idx, _ in enumerate(self.src_datasets):
            dm = torch.where(dset_mask == dset_idx)[0]
            if dm.size(0) == 0:
                continue
            dset2sp = torch.unique(p2sp[dm])
            k_dset = k[dset2sp]
            q_dset = q[dset2sp]
            mask = torch.where(k_dset[:, 0] != 0)[0]
            k_dset = k_dset[mask]
            q_dset = q_dset[mask]
            # nce += self.nce(k_dset, q_dset)
            nce.append(self.nce(k_dset, q_dset))

        # return nce
        return torch.mean(torch.stack(nce))

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

        # return self.supcon(k, q, sp_l), self.supcon_intra(k, k, sp_l)
        return self.supcon(k, q, sp_l) + self.supcon_intra(k, k, sp_l)
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
    #             (self.superpixel_size * superpixels.size(0), pairing_points.shape[0])
    #         )
    #
    #         one_hot_I = torch.sparse_coo_tensor(
    #             torch.stack((
    #                 superpixels_I, idx_I
    #             ), dim=0),
    #             torch.ones(total_pixels, device=superpixels.device),
    #             (self.superpixel_size * superpixels.size(0), total_pixels)
    #         )
    #
    #     k = one_hot_P @ output_points[pairing_points]
    #     k = k / (torch.sparse.sum(one_hot_P, 1).to_dense()[:, None] + 1e-6)
    #     q = one_hot_I @ output_images.permute(0, 2, 3, 1).flatten(0, 2)
    #     q = q / (torch.sparse.sum(one_hot_I, 1).to_dense()[:, None] + 1e-6)
    #
    #     sp_l = torch.arange(0, k.size(0), device=self.device) % self.superpixel_size
    #     mask = torch.where(torch.logical_and(k[:, 0] != 0, sp_l != self.ignored_label))
    #     k = k[mask]
    #     q = q[mask]
    #     sp_l = sp_l[mask]
    #
    #     return self.criterion(k, q, sp_l)

    # SEAL+SEEM
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
    #     # k = k[1:, :]
    #     # q = q[1:, :]
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

    # def validation_step(self, batch, batch_idx):
    #     sparse_input = ME.SparseTensor(batch["sinput_F"], batch["sinput_C"])
    #     output_points = self.model_points(sparse_input).F
    #     self.model_images.eval()
    #     output_images = self.model_images(batch["input_I"])
    #
    #     losses = [
    #         getattr(self, loss)(batch, output_points, output_images)
    #         for loss in self.losses
    #     ]
    #     loss = torch.mean(torch.stack(losses))
    #     # p2c_loss =  self.loss_p2c_maxpool(batch, output_points)
    #     # loss += p2c_loss
    #     self.val_losses.append(loss.detach().cpu())
    #
    #     self.log(
    #         "val_loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=self.batch_size
    #     )
    #     # self.log(
    #     #     "p2c_loss", p2c_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size
    #     # )
    #     return loss
    def validation_step(self, batch, batch_idx):
        if self.decoupled_head:
            sparse_input = ME.SparseTensor(batch["sinput_F"], batch["sinput_C"])
            output_points, output_points_tmp = self.model_points(sparse_input)
            self.model_images.eval()
            output_images, output_images_tmp = self.model_images(batch["input_I"])
            # each loss is applied independtly on each GPU
            # losses = [
            #     self.loss_superpixels_average(batch, output_points, output_images),
            #     self.loss_nce(batch, output_points_tmp, output_images_tmp),
            # ]
            # loss = torch.mean(torch.stack(losses))
            # nce_loss = self.loss_nce_with_conditions(batch, output_points_tmp, output_images_tmp)
            nce_loss, nce_loss_intra = self.loss_nce(batch, output_points_tmp, output_images_tmp)
            supcon_loss, supcon_intra_loss = self.loss_superpixels_average(batch, output_points, output_images)
            # loss = (nce_loss + supcon_loss) / 2 + (supcon_intra_loss + nce_loss_intra) / 2
            loss = nce_loss + supcon_loss + nce_loss_intra + supcon_intra_loss
            # loss = (nce_loss + supcon_loss + supcon_intra_loss) / 3
            self.log('val_nce_loss', nce_loss, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
            self.log('val_nce_loss_intra', nce_loss_intra, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
            self.log(
                "val_loss", loss, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size
            )
            self.train_losses.append(loss.detach().cpu())
            return loss
        else:
            sparse_input = ME.SparseTensor(batch["sinput_F"], batch["sinput_C"])
            output_points = self.model_points(sparse_input).F
            self.model_images.eval()
            output_images = self.model_images(batch["input_I"])

            losses = [
                getattr(self, loss)(batch, output_points, output_images)
                for loss in self.losses
            ]
            loss = torch.mean(torch.stack(losses))
            # nce_loss = self.loss_nce(batch, output_points, output_images)
            # supcon_loss, supcon_intra_loss = self.loss_superpixels_average(batch, output_points, output_images)
            # loss = nce_loss + supcon_loss + supcon_intra_loss
            # loss = (nce_loss + supcon_loss + supcon_intra_loss) / 3
            self.val_losses.append(loss.detach().cpu())
            # self.log('val_nce_loss', nce_loss, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
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

# v2
# import os
# import re
# import torch
# import numpy as np
# import torch.optim as optim
# import MinkowskiEngine as ME
# # import pytorch_lightning as pl
# import lightning.pytorch as pl
# from pretrain.criterion import NCELoss, SupConLoss, SupConLossIntra
# # from pytorch_lightning.utilities import rank_zero_only
# from lightning.pytorch.utilities import rank_zero_only
#
#
# class LightningPretrain(pl.LightningModule):
#     def __init__(self, model_points, model_images, config):
#         super().__init__()
#         self.model_points = model_points
#         self.model_images = model_images
#         self._config = config
#         self.losses = config["losses"]
#         self.train_losses = []
#         self.val_losses = []
#         self.num_matches = config["num_matches"]
#         self.batch_size = config["batch_size"]
#         self.num_epochs = config["num_epochs"]
#         self.superpixel_size = config["superpixel_size"]
#         self.epoch = 0
#         if config["resume_path"] is not None:
#             self.epoch = int(
#                 re.search(r"(?<=epoch=)[0-9]+", config["resume_path"])[0]
#             )
#         # self.criterion = NCELoss(temperature=config["NCE_temperature"])
#         self.criterion = SupConLoss(temperature=config["NCE_temperature"])
#         self.supcon_intra = SupConLossIntra(temperature=config['NCE_temperature'])
#         self.working_dir = os.path.join(config["working_dir"], str(config["datetime"]) + '-' + str(config['run_dir']))
#         if os.environ.get("LOCAL_RANK", 0) == 0:
#             os.makedirs(self.working_dir, exist_ok=True)
#
#     def configure_optimizers(self):
#         optimizer = optim.SGD(
#             list(self.model_points.parameters()) + list(self.model_images.parameters()),
#             lr=self._config["lr"],
#             momentum=self._config["sgd_momentum"],
#             dampening=self._config["sgd_dampening"],
#             weight_decay=self._config["weight_decay"],
#         )
#         scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.num_epochs)
#         return [optimizer], [scheduler]
#
#     # def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
#     #     optimizer.zero_grad(set_to_none=True)
#
#     def training_step(self, batch, batch_idx):
#         sparse_input = ME.SparseTensor(batch["sinput_F"], batch["sinput_C"])
#         output_points = self.model_points(sparse_input).F
#         self.model_images.eval()
#         self.model_images.decoder.train()
#         output_images = self.model_images(batch["input_I"])
#         # each loss is applied independtly on each GPU
#         losses = [
#             getattr(self, loss)(batch, output_points, output_images)
#             for loss in self.losses
#         ]
#         loss = torch.mean(torch.stack(losses))
#         # torch.cuda.empty_cache()
#         self.log(
#             "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size
#         )
#         self.train_losses.append(loss.detach().cpu())
#         return loss
#
#     def loss(self, batch, output_points, output_images):
#         pairing_points = batch["pairing_points"]
#         pairing_images = batch["pairing_images"]
#         idx = np.random.choice(pairing_points.shape[0], self.num_matches, replace=False)
#         k = output_points[pairing_points[idx]]
#         m = tuple(pairing_images[idx].T.long())
#         q = output_images.permute(0, 2, 3, 1)[m]
#         return self.criterion(k, q)
#
#     def loss_superpixels_average(self, batch, output_points, output_images):
#         # compute a superpoints to superpixels loss using superpixels
#         # torch.cuda.empty_cache()  # This method is extremely memory intensive
#         superpixels = batch["superpixels"].long()  # [batch_size*6, H, W,]
#         pairing_images = batch["pairing_images"]  # 每个点对应的像素, [N_fov, 3], (图片idx, x, y)
#         pairing_points = batch["pairing_points"]  # 有对应像素的点的idx, [N_fov,]
#
#         # max_sp = 0
#         # max_per_sp = 0
#         # for sp_i, sp in enumerate(superpixels):
#         #     max_per_sp = torch.max(sp) if torch.max(sp) > max_per_sp else max_per_sp
#         #     v_m = (sp != 0)
#         #     sp[v_m] = sp[v_m] + max_sp
#         #     max_sp = torch.max(sp)
#         #     superpixels[sp_i] = sp
#         superpixels = (
#             torch.arange(
#                 0,
#                 output_images.shape[0] * self.superpixel_size,
#                 self.superpixel_size,
#                 device=self.device,
#             )[:, None, None] + superpixels
#         )
#
#         m = tuple(pairing_images.cpu().T.long())
#
#         superpixels_I = superpixels.flatten()
#         idx_P = torch.arange(pairing_points.shape[0], device=superpixels.device)
#         total_pixels = superpixels_I.shape[0]
#         idx_I = torch.arange(total_pixels, device=superpixels.device)
#
#         with torch.no_grad():
#             one_hot_P = torch.sparse_coo_tensor(
#                 torch.stack((
#                     superpixels[m], idx_P  # 超像素id, 点id的id
#                 ), dim=0),
#                 torch.ones(pairing_points.shape[0], device=superpixels.device),
#                 (self.superpixel_size * superpixels.size(0), pairing_points.shape[0])
#             )
#
#             one_hot_I = torch.sparse_coo_tensor(
#                 torch.stack((
#                     superpixels_I, idx_I
#                 ), dim=0),
#                 torch.ones(total_pixels, device=superpixels.device),
#                 (self.superpixel_size * superpixels.size(0), total_pixels)
#             )
#
#         k = one_hot_P @ output_points[pairing_points]
#         k = k / (torch.sparse.sum(one_hot_P, 1).to_dense()[:, None] + 1e-6)
#         q = one_hot_I @ output_images.permute(0, 2, 3, 1).flatten(0, 2)
#         q = q / (torch.sparse.sum(one_hot_I, 1).to_dense()[:, None] + 1e-6)
#
#         sp_l = torch.arange(0, k.size(0), device=self.device) % self.superpixel_size
#         mask = torch.where(torch.logical_and(k[:, 0] != 0, sp_l != 0))
#         k = k[mask]
#         q = q[mask]
#         sp_l = sp_l[mask]
#
#         return self.criterion(k, q, sp_l) + self.supcon_intra(k, k, sp_l)
#
#     # def loss_superpixels_average(self, batch, output_points, output_images):
#     #     # compute a superpoints to superpixels loss using superpixels
#     #     # torch.cuda.empty_cache()  # This method is extremely memory intensive
#     #     superpixels = batch["superpixels"].long()  # [batch_size*6, H, W,]
#     #     pairing_images = batch["pairing_images"]  # 每个点对应的像素, [N_fov, 3], (图片idx, x, y)
#     #     pairing_points = batch["pairing_points"]  # 有对应像素的点的idx, [N_fov,]
#     #
#     #     max_sp = 0
#     #     max_per_sp = 0
#     #     for sp_i, sp in enumerate(superpixels):
#     #         max_per_sp = torch.max(sp) if torch.max(sp) > max_per_sp else max_per_sp
#     #         v_m = (sp != 0)
#     #         sp[v_m] = sp[v_m] + max_sp
#     #         max_sp = torch.max(sp)
#     #         superpixels[sp_i] = sp
#     #     # superpixels = (
#     #     #     torch.arange(
#     #     #         0,
#     #     #         output_images.shape[0] * self.superpixel_size,
#     #     #         self.superpixel_size,
#     #     #         device=self.device,
#     #     #     )[:, None, None] + superpixels
#     #     # )
#     #     m = tuple(pairing_images.cpu().T.long())
#     #
#     #     superpixels_I = superpixels.flatten()
#     #     # v_m = superpixels != 0
#     #     # superpixels_I = superpixels_I[v_m]
#     #     idx_P = torch.arange(pairing_points.shape[0], device=superpixels.device)
#     #     total_pixels = superpixels_I.shape[0]
#     #     idx_I = torch.arange(total_pixels, device=superpixels.device)
#     #
#     #     with torch.no_grad():
#     #         # one_hot_P = torch.sparse_coo_tensor(
#     #         #     torch.stack((
#     #         #         superpixels[m], idx_P  # 超像素id, 点id的id
#     #         #     ), dim=0),
#     #         #     torch.ones(pairing_points.shape[0], device=superpixels.device),
#     #         #     (superpixels.shape[0] * self.superpixel_size, pairing_points.shape[0])
#     #         # )
#     #         #
#     #         # one_hot_I = torch.sparse_coo_tensor(
#     #         #     torch.stack((
#     #         #         superpixels_I, idx_I
#     #         #     ), dim=0),
#     #         #     torch.ones(total_pixels, device=superpixels.device),
#     #         #     (superpixels.shape[0] * self.superpixel_size, total_pixels)
#     #         # )
#     #         # one_hot_P = torch.sparse_coo_tensor(
#     #         #     torch.stack((
#     #         #         superpixels[m], idx_P  # 超像素id, 点id的id
#     #         #     ), dim=0),
#     #         #     torch.ones(pairing_points.shape[0], device=superpixels.device),
#     #         #     (max_sp + 1, pairing_points.shape[0])
#     #         # )
#     #         #
#     #         # one_hot_I = torch.sparse_coo_tensor(
#     #         #     torch.stack((
#     #         #         superpixels_I, idx_I
#     #         #     ), dim=0),
#     #         #     torch.ones(total_pixels, device=superpixels.device),
#     #         #     (max_sp + 1, total_pixels)
#     #         # )
#     #         one_hot_P = torch.sparse_coo_tensor(
#     #             torch.stack((
#     #                 superpixels[m], idx_P  # 超像素id, 点id的id
#     #             ), dim=0),
#     #             torch.ones(pairing_points.shape[0], device=superpixels.device),
#     #             (max_per_sp * superpixels.size(0) + 1, pairing_points.shape[0])
#     #         )
#     #
#     #         one_hot_I = torch.sparse_coo_tensor(
#     #             torch.stack((
#     #                 superpixels_I, idx_I
#     #             ), dim=0),
#     #             torch.ones(total_pixels, device=superpixels.device),
#     #             (max_per_sp * superpixels.size(0) + 1, total_pixels)
#     #         )
#     #
#     #     k = one_hot_P @ output_points[pairing_points]
#     #     k = k / (torch.sparse.sum(one_hot_P, 1).to_dense()[:, None] + 1e-6)
#     #     q = one_hot_I @ output_images.permute(0, 2, 3, 1).flatten(0, 2)
#     #     q = q / (torch.sparse.sum(one_hot_I, 1).to_dense()[:, None] + 1e-6)
#     #
#     #     k = k[1:, :]
#     #     q = q[1:, :]
#     #     mask = torch.where(k[:, 0] != 0)
#     #     k = k[mask]
#     #     q = q[mask]
#     #
#     #     return self.criterion(k, q)
#
#     # def training_epoch_end(self, outputs):
#     #     self.epoch += 1
#     #     if self.epoch == self.num_epochs:
#     #         self.save()
#     #     return super().training_epoch_end(outputs)
#
#     def on_train_epoch_end(self) -> None:
#         self.epoch += 1
#         if self.epoch == self.num_epochs:
#             self.save()
#
#     def validation_step(self, batch, batch_idx):
#         sparse_input = ME.SparseTensor(batch["sinput_F"], batch["sinput_C"])
#         output_points = self.model_points(sparse_input).F
#         self.model_images.eval()
#         output_images = self.model_images(batch["input_I"])
#
#         losses = [
#             getattr(self, loss)(batch, output_points, output_images)
#             for loss in self.losses
#         ]
#         loss = torch.mean(torch.stack(losses))
#         self.val_losses.append(loss.detach().cpu())
#
#         self.log(
#             "val_loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=self.batch_size
#         )
#         return loss
#
#     @rank_zero_only
#     def save(self):
#         path = os.path.join(self.working_dir, "model.pt")
#         torch.save(
#             {
#                 "model_points": self.model_points.state_dict(),
#                 "model_images": self.model_images.state_dict(),
#                 "epoch": self.epoch,
#                 "config": self._config,
#             },
#             path,
#         )
