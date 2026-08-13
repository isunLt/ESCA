import torch
import random
import numpy as np
from torchvision.transforms import InterpolationMode
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms.functional import resize, resized_crop, hflip
# from visualize_utils import visualize_img


class ComposeClouds:
    """
    Compose multiple transformations on a point cloud.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, pc):
        for transform in self.transforms:
            pc = transform(pc)
        return pc


class Rotation_z:
    """
    Random rotation of a point cloud around the z axis.
    """

    def __init__(self):
        pass

    def __call__(self, pc):
        angle = np.random.random() * 2 * np.pi
        c = np.cos(angle)
        s = np.sin(angle)
        R = torch.tensor(
            [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32
        )
        pc = pc @ R.T
        return pc


class FlipAxis:
    """
    Flip a point cloud in the x and/or y axis, with probability p for each.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, pc):
        for curr_ax in range(2):
            if random.random() < self.p:
                pc[:, curr_ax] = -pc[:, curr_ax]
        return pc


def make_transforms_clouds(config):
    """
    Read the config file and return the desired transformation on point clouds.
    """
    transforms = []
    if config["transforms_clouds"] is not None:
        for t in config["transforms_clouds"]:
            if t.lower() == "rotation":
                transforms.append(Rotation_z())
            elif t.lower() == "flipaxis":
                transforms.append(FlipAxis())
            else:
                raise Exception(f"Unknown transformation: {t}")
    if not len(transforms):
        return None
    return ComposeClouds(transforms)


class ComposeAsymmetrical:
    """
    Compose multiple transformations on a point cloud, and image and the
    intricate pairings between both (only available for the heavy dataset).
    Note: Those transformations have the ability to increase the number of
    images, and drastically modify the pairings
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, pc, features, img, pairing_points, pairing_images, superpixels=None, points_labels=None):
        for transform in self.transforms:
            pc, features, img, pairing_points, pairing_images, superpixels, points_labels = transform(
                pc, features, img, pairing_points, pairing_images, superpixels, points_labels
            )
        ret_list = [pc, features, img, pairing_points, pairing_images]
        if superpixels is not None:
            ret_list += [superpixels]
        if points_labels is not None:
            ret_list += [points_labels]
        return ret_list
        # return pc, features, img, pairing_points, pairing_images, superpixels, points_labels


class ResizedCrop:
    """
    Resize and crop an image, and adapt the pairings accordingly.
    """

    def __init__(
        self,
        image_crop_size=(224, 416),
        image_crop_range=[0.3, 1.0],
        image_crop_ratio=(14.0 / 9.0, 17.0 / 9.0),
        image_interpolation=InterpolationMode.BILINEAR,
        crop_center=False,
    ):
        self.crop_size = image_crop_size
        self.crop_range = image_crop_range
        self.crop_ratio = image_crop_ratio
        self.img_interpolation = image_interpolation
        self.crop_center = crop_center

    def __call__(self, pc, features, images, pairing_points, pairing_images, superpixels, point_labels):
        # print('ResizedCrop Start')
        # imgs = torch.empty(
        #     (images.shape[0], 3) + tuple(self.crop_size), dtype=torch.float32
        # )
        imgs = torch.empty(
            (len(images), 3) + tuple(self.crop_size), dtype=torch.float32
        )
        if superpixels is not None:
            if superpixels[0].ndim == 2:
                superpixels = superpixels.unsqueeze(1)
            sps = torch.empty(
                (len(images),) + tuple(self.crop_size), dtype=torch.uint8
            )
            # sps = torch.empty(
            #     (images.shape[0],) + tuple(self.crop_size), dtype=torch.uint8
            # )
        pairing_points_out = np.empty(0, dtype=np.int64)
        pairing_images_out = np.empty((0, 3), dtype=np.int64)
        if self.crop_center:
            pairing_points_out = pairing_points
            if isinstance(images, list):
                _, h, w = images[0].shape
            else:
                _, _, h, w = images.shape
            for id, img in enumerate(images):
                mask = pairing_images[:, 0] == id
                p2 = pairing_images[mask]
                p2 = np.round(
                    np.multiply(p2, [1.0, self.crop_size[0] / h, self.crop_size[1] / w])
                ).astype(np.int64)

                imgs[id] = resize(img, self.crop_size, self.img_interpolation, antialias=None)
                if superpixels is not None:
                    sps[id] = resize(
                        superpixels[id], self.crop_size, InterpolationMode.NEAREST, antialias=None
                    )

                p2[:, 1] = np.clip(0, self.crop_size[0] - 1, p2[:, 1])
                p2[:, 2] = np.clip(0, self.crop_size[1] - 1, p2[:, 2])
                pairing_images_out = np.concatenate((pairing_images_out, p2))

        else:
            for id, img in enumerate(images):
                successfull = False
                mask = pairing_images[:, 0] == id
                P1 = pairing_points[mask]
                P2 = pairing_images[mask]  # (图像id, h, w)
                flag = False
                if len(P1) == 0:
                    flag = True
                # cnt = 0
                while not successfull:
                    i, j, h, w = RandomResizedCrop.get_params(
                        img, self.crop_range, self.crop_ratio
                    )
                    p1 = P1.copy()
                    p2 = P2.copy()
                    p2 = np.round(
                        np.multiply(
                            p2 - [0, i, j],
                            [1.0, self.crop_size[0] / h, self.crop_size[1] / w],
                        )
                    ).astype(np.int64)

                    valid_indexes_0 = np.logical_and(
                        p2[:, 1] < self.crop_size[0], p2[:, 1] >= 0
                    )
                    valid_indexes_1 = np.logical_and(
                        p2[:, 2] < self.crop_size[1], p2[:, 2] >= 0
                    )
                    valid_indexes = np.logical_and(valid_indexes_0, valid_indexes_1)
                    sum_indexes = valid_indexes.sum()
                    len_indexes = len(valid_indexes)
                    if flag:
                        break
                    if len_indexes == 0:
                        continue
                    # cnt += 1
                    if sum_indexes >  1024 or sum_indexes / len_indexes > 0.75:  # 裁剪后的图像必须保留大于1024或者大于75%的该图像上的点，否则重新裁剪
                        successfull = True

                imgs[id] = resized_crop(
                    img, i, j, h, w, self.crop_size, self.img_interpolation, antialias=None
                )  # 以[i,j]为原点，(h, w)为高和宽对图像进行裁剪，将裁剪后的图像resize到self.crop_size大小
                if superpixels is not None:
                    sps[id] = resized_crop(
                        superpixels[id],
                        i,
                        j,
                        h,
                        w,
                        self.crop_size,
                        InterpolationMode.NEAREST,
                        antialias=None
                    )
                pairing_points_out = np.concatenate(
                    (pairing_points_out, p1[valid_indexes])
                )
                pairing_images_out = np.concatenate(
                    (pairing_images_out, p2[valid_indexes])
                )
        # print('ResizedCrop End')
        if superpixels is None:
            return pc, features, imgs, pairing_points_out, pairing_images_out, superpixels, point_labels
        return pc, features, imgs, pairing_points_out, pairing_images_out, sps, point_labels


class FlipHorizontal:
    """
    Flip horizontaly the image with probability p and adapt the matching accordingly.
    """

    def __init__(self, p=0.5):
        self.p = p  # default 0.5

    def __call__(self, pc, features, images, pairing_points, pairing_images, superpixels, point_labels):
        # print('FlipHorizontal Start')
        w = images.shape[3]
        for i, img in enumerate(images):
            if random.random() < self.p:  # 有50%概率对图像进行水平翻转
                images[i] = hflip(img)
                if superpixels is not None:
                    superpixels[i] = hflip(superpixels[i: i + 1])
                mask = pairing_images[:, 0] == i
                pairing_images[mask, 2] = w - 1 - pairing_images[mask, 2]
        # print('FlipHorizontal End')
        return pc, features, images, pairing_points, pairing_images, superpixels, point_labels


class DropCuboids:
    """
    Drop random cuboids in a cloud
    """

    def __call__(self, pc, features, images, pairing_points, pairing_images, superpixels, point_labels):
        # print('DropCuboids Start')
        range_xyz = torch.max(pc, axis=0)[0] - torch.min(pc, axis=0)[0]

        crop_range = np.random.random() * 0.2
        new_range = range_xyz * crop_range / 2.0

        sample_center = pc[np.random.choice(len(pc))]

        max_xyz = sample_center + new_range
        min_xyz = sample_center - new_range

        upper_idx = torch.sum((pc[:, 0:3] < max_xyz).to(torch.int32), 1) == 3
        lower_idx = torch.sum((pc[:, 0:3] > min_xyz).to(torch.int32), 1) == 3

        new_pointidx = ~((upper_idx) & (lower_idx))  # 要drop的cube之外的点，也就是要保留的点
        pc_out = pc[new_pointidx]
        features_out = features[new_pointidx]
        if point_labels is not None:
            point_labels_out = point_labels[new_pointidx]
        else:
            point_labels_out = None

        mask = new_pointidx[pairing_points]  # 要保留的点中，在相机视野内的点
        cs = torch.cumsum(new_pointidx, 0) - 1
        pairing_points_out = pairing_points[mask]
        pairing_points_out = cs[pairing_points_out]  # 要保留点的idx
        pairing_images_out = pairing_images[mask]  # 要保留点对应的像素坐标

        successfull = True
        for id in range(len(images)):
            if np.sum(pairing_images_out[:, 0] == id) < 1024:
                successfull = False  # 如果drop掉一个cube之后某帧图像对应的点云点数小于1024，则认为其不满足要求

        if successfull:  # 满足要求，使用drop之后的点云
            return (
                pc_out,
                features_out,
                images,
                np.array(pairing_points_out),
                np.array(pairing_images_out),
                superpixels,
                point_labels_out
            )
        return pc, features, images, pairing_points, pairing_images, superpixels, point_labels  # 不满足要求，则用原点云


def make_transforms_asymmetrical(config):
    """
    Read the config file and return the desired mixed transformation.
    """
    transforms = []
    if config["transforms_mixed"] is not None:
        for t in config["transforms_mixed"]:
            if t.lower() == "resizedcrop":
                transforms.append(
                    ResizedCrop(
                        image_crop_size=config["crop_size"],
                        image_crop_ratio=config["crop_ratio"],
                    )
                )
            elif t.lower() == "fliphorizontal":
                transforms.append(FlipHorizontal())
            elif t.lower() == "dropcuboids":
                transforms.append(DropCuboids())
            else:
                raise Exception(f"Unknown transformation {t}")
    if not len(transforms):
        return None
    return ComposeAsymmetrical(transforms)


def make_transforms_asymmetrical_val(config):
    """
    Read the config file and return the desired mixed transformation
    for the validation only.
    """
    transforms = []
    if config["transforms_mixed"] is not None:
        for t in config["transforms_mixed"]:
            if t.lower() == "resizedcrop":
                transforms.append(
                    ResizedCrop(image_crop_size=config["crop_size"], crop_center=True)
                )
    if not len(transforms):
        return None
    return ComposeAsymmetrical(transforms)
