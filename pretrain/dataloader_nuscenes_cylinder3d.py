import os
import copy
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from spconv.pytorch.utils import PointToVoxel
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.geometry_utils import view_points
import pickle

CUSTOM_SPLIT = [
    "scene-0008", "scene-0009", "scene-0019", "scene-0029", "scene-0032", "scene-0042",
    "scene-0045", "scene-0049", "scene-0052", "scene-0054", "scene-0056", "scene-0066",
    "scene-0067", "scene-0073", "scene-0131", "scene-0152", "scene-0166", "scene-0168",
    "scene-0183", "scene-0190", "scene-0194", "scene-0208", "scene-0210", "scene-0211",
    "scene-0241", "scene-0243", "scene-0248", "scene-0259", "scene-0260", "scene-0261",
    "scene-0287", "scene-0292", "scene-0297", "scene-0305", "scene-0306", "scene-0350",
    "scene-0352", "scene-0358", "scene-0361", "scene-0365", "scene-0368", "scene-0377",
    "scene-0388", "scene-0391", "scene-0395", "scene-0413", "scene-0427", "scene-0428",
    "scene-0438", "scene-0444", "scene-0452", "scene-0453", "scene-0459", "scene-0463",
    "scene-0464", "scene-0475", "scene-0513", "scene-0533", "scene-0544", "scene-0575",
    "scene-0587", "scene-0589", "scene-0642", "scene-0652", "scene-0658", "scene-0669",
    "scene-0678", "scene-0687", "scene-0701", "scene-0703", "scene-0706", "scene-0710",
    "scene-0715", "scene-0726", "scene-0735", "scene-0740", "scene-0758", "scene-0786",
    "scene-0790", "scene-0804", "scene-0806", "scene-0847", "scene-0856", "scene-0868",
    "scene-0882", "scene-0897", "scene-0899", "scene-0976", "scene-0996", "scene-1012",
    "scene-1015", "scene-1016", "scene-1018", "scene-1020", "scene-1024", "scene-1044",
    "scene-1058", "scene-1094", "scene-1098", "scene-1107",
]

def cylinder3d_custom_collate_fn(list_data):
    """
    Custom collate function adapted for creating batches with spconv.
    """
    (
        pc_features,
        coords,
        images,
        pairing_points,
        pairing_images,
        superpixels,
    ) = list(zip(*list_data))

    offset = 0
    temp_pairing_points = []
    for batch_id in range(len(coords)):
        # Move batch ids to the beginning
        # coords[batch_id][:, 0] = batch_id
        # pairing_points[batch_id][:] += offset
        pairing_images[batch_id][:, 0] += batch_id * images[0].shape[0]
        pc_ind = coords[batch_id]  # (N_pc,3)
        points = pairing_points[batch_id]  # (N_)
        unq, unq_inv = np.unique(pc_ind, return_inverse=True, axis=0)  # unq_inv: point的 voxel_id
        points = unq_inv[points]  # 视野内的点对应的voxel id
        points = points + offset
        offset = offset + len(unq)
        temp_pairing_points.append(points)

        # offset += coords[batch_id].shape[0]

    pc_features = [torch.Tensor(data) for data in pc_features]
    coords = [torch.Tensor(data) for data in coords]
    images_batch = torch.cat(images, 0).float()
    superpixels_batch = torch.tensor(np.concatenate(superpixels))
    pairing_points = torch.tensor(np.concatenate(temp_pairing_points))
    pairing_images = torch.tensor(np.concatenate(pairing_images))
    return {
        "pc_features": pc_features,
        "pc_grid_index": coords,
        "input_I": images_batch,
        "pairing_points": pairing_points,
        "pairing_images": pairing_images,
        "superpixels": superpixels_batch,
    }


class NuScenesDatasetCylinder3d(Dataset):
    """
    Dataset returning a lidar scene and associated labels.
    """

    def __init__(self,
                 phase,
                 config,
                 shuffle=False,
                 cloud_transforms=None,
                 mixed_transforms=None,
                 **kwargs,
                 ):
        self.phase = phase
        self.shuffle = shuffle
        self.cloud_transforms = cloud_transforms
        self.mixed_transforms = mixed_transforms
        self.config = config
        self.use_polar = config['use_polar'] if 'use_polar' in config else False
        dataset_name = config["dataset"]
        assert self.use_polar
        if dataset_name == "nuscenes":
            if self.use_polar:
                # 极坐标系下
                self.point_cloud_range = np.array([0, -3.1415926, -3, 50, 3.1415926, 3], dtype=np.float32)
                self.grid_size = [480, 360, 32]
                # self.voxel_size: array([0.10416667, 0.01745329, 0.1875    ])
                self.voxel_size = (self.point_cloud_range[3:] - self.point_cloud_range[0:3]) / self.grid_size
                MAX_POINTS_PER_VOXEL = 10  # nuScenes
                MAX_NUMBER_OF_VOXELS = 100000  # nuScenes
                self._voxel_generator = PointToVoxel(
                    vsize_xyz=self.voxel_size,
                    coors_range_xyz=self.point_cloud_range,
                    num_point_features=4,
                    max_num_points_per_voxel=MAX_POINTS_PER_VOXEL,
                    max_num_voxels=MAX_NUMBER_OF_VOXELS
                )
            else:
                # 笛卡尔坐标系
                self.voxel_size = [0.1, 0.1, 0.2]  # nuScenes
                self.point_cloud_range = np.array([-51.2, -51.2, -5.0, 51.2, 51.2, 3.0], dtype=np.float32)  # nuScenes
                #
                MAX_POINTS_PER_VOXEL = 10  # nuScenes
                MAX_NUMBER_OF_VOXELS = 60000  # nuScenes
                self._voxel_generator = PointToVoxel(
                    vsize_xyz=self.voxel_size,
                    coors_range_xyz=self.point_cloud_range,
                    num_point_features=4,
                    max_num_points_per_voxel=MAX_POINTS_PER_VOXEL,
                    max_num_voxels=MAX_NUMBER_OF_VOXELS
                )
                self.grid_size = self._voxel_generator.grid_size
            self.max_bound = self.point_cloud_range[3:]
            self.min_bound = self.point_cloud_range[0:3]
        else:
            raise Exception("Dataset unknown")

        self.superpixels_type = config["superpixels_type"]
        self.num_point_features = 4
        ################
        if "cached_nuscenes" in kwargs:
            self.nusc = kwargs["cached_nuscenes"]
        else:
            self.nusc = NuScenes(
                version="v1.0-trainval", dataroot="datasets/nuscenes", verbose=False
            )

        self.list_keyframes = []
        # a skip ratio can be used to reduce the dataset size and accelerate experiments
        try:
            skip_ratio = config["dataset_skip_step"]
        except KeyError:
            skip_ratio = 1
        skip_counter = 0

        # assert phase in ('train', 'parametrizing')
        #
        # phase_scenes = create_splits_scenes()['train']
        if phase in ("train", "val", "test"):
            phase_scenes = create_splits_scenes()[phase]
        elif phase == "parametrizing":
            phase_scenes = list(
                set(create_splits_scenes()["train"]) - set(CUSTOM_SPLIT)
            )
        elif phase == "verifying":
            phase_scenes = CUSTOM_SPLIT

        for scene_idx in range(len(self.nusc.scene)):
            scene = self.nusc.scene[scene_idx]
            if scene["name"] in phase_scenes:
                skip_counter += 1
                if skip_counter % skip_ratio == 0:
                    self.create_list_of_scans(scene)

        self.superpixels_path = config['superpixels_path']
        self.ignored_label = config['ignored_label']
        self.normalize_intensity = config['normalize_intensity']
        self.use_vse = config['use_vse']

        if self.use_vse and phase in ['parametrizing', 'train']:
            sweeps_skip_step = config["sweeps_skip_step"]
            sweeps_pair_list_path = config['sweeps_pair_list_path']
            with open(sweeps_pair_list_path, 'rb') as f:
                sweeps_pair_list = pickle.load(f)
            self.list_keyframes.extend(sweeps_pair_list['top_1'][::sweeps_skip_step])

    def create_list_of_scans(self, scene):
        # Get first and last keyframe in the scene
        current_sample_token = scene["first_sample_token"]

        # Loop to get all successive keyframes
        list_data = []
        while current_sample_token != "":
            current_sample = self.nusc.get("sample", current_sample_token)
            list_data.append(current_sample["data"])
            current_sample_token = current_sample["next"]

        # Add new scans in the list
        self.list_keyframes.extend(list_data)

    def map_pointcloud_to_image(self, data, min_dist: float = 1.0):
        """
        Given a lidar token and camera sample_data token, load pointcloud and map it to
        the image plane. Code adapted from nuscenes-devkit
        https://github.com/nutonomy/nuscenes-devkit.
        :param min_dist: Distance from the camera below which points are discarded.
        """
        pointsensor = self.nusc.get("sample_data", data["LIDAR_TOP"])
        pcl_path = os.path.join(self.nusc.dataroot, pointsensor["filename"])
        pc_original = LidarPointCloud.from_file(pcl_path)
        pc = pc_original.points
        dist = pc[0] * pc[0] + pc[1] * pc[1]
        mask = (dist <= 2621.44) & \
               (pc[2] >= self.point_cloud_range[2]) & \
               (pc[2] <= self.point_cloud_range[5])
        pc_original = LidarPointCloud(pc[:, mask])
        pc_ref = pc_original.points

        images = []
        superpixels = []
        img_path_list = []
        pairing_points = np.empty(0, dtype=np.int64)
        pairing_images = np.empty((0, 3), dtype=np.int64)
        camera_list = [
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK_RIGHT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_FRONT_LEFT",
        ]
        if self.shuffle:
            np.random.shuffle(camera_list)

        for i, camera_name in enumerate(camera_list):
            pc = copy.deepcopy(pc_original)
            cam = self.nusc.get("sample_data", data[camera_name])
            img_path = os.path.join(self.nusc.dataroot, cam["filename"])
            img_path_list.append(img_path)
            im = np.array(Image.open(img_path))
            # if self.superpixels_type != 'dinov2':
            #     sp = Image.open(
            #         f"superpixels/nuscenes/"
            #         f"superpixels_{self.superpixels_type}/{cam['token']}.png"
            #     )
            #     sp = np.array(sp)
            # else:
            #     sp = np.fromfile(
            #         f"superpixels/nuscenes/superpixels_dinov2/{cam['token']}_dino_mask.bin",
            #         dtype=np.uint8
            #     )
            #     sp = np.reshape(sp, (900, 1600))

            if self.superpixels_type == 'dinov2':
                if pointsensor['is_key_frame']:
                    sp_path = os.path.join(self.superpixels_path, 'keyframes', str(cam['token']) + '_dino_mask.bin')
                else:
                    sp_path = os.path.join(self.superpixels_path, 'sweeps', str(cam['token']) + '_dino_mask.bin')
                if os.path.exists(sp_path):
                    sp = np.fromfile(sp_path, dtype=np.uint8).reshape(900, 1600).astype(np.int32)
                else:
                    sp = np.array(Image.open(os.path.join(self.superpixels_path.replace('dino_mask', 'dino_mask_png'), 'keyframes', str(cam['token']) + '.png')))

                # sp_path = os.path.join(self.superpixels_path, 'keyframes', str(cam['token'])+'_dino_mask.bin')
                # if os.path.exists(sp_path):
                #     sp = np.fromfile(sp_path, dtype=np.uint8).reshape(900, 1600).astype(np.int32)
                # else:
                #     sp = np.array(Image.open(os.path.join(self.superpixels_path.replace('dino_mask', 'dino_mask_png'), 'keyframes', str(cam['token']) + '.png')))

                # sp = np.fromfile(os.path.join(self.superpixels_path, 'keyframes', str(cam['token'])+'_dino_mask.bin'), dtype=np.uint8).reshape(900, 1600).astype(np.int32)
                # sp = np.array(Image.open(os.path.join(self.superpixels_path, 'keyframes', str(cam['token']) + '.png')))
            elif self.superpixels_type == 'seem':
                sp = np.array(Image.open(os.path.join(self.superpixels_path, str(cam['token']) + '.png')))
            elif self.superpixels_type == 'oneformer':
                sp = np.array(Image.open(os.path.join(self.superpixels_path, str(cam['token']) + '.png')))
            elif self.superpixels_type == 'san':
                sp = np.array(Image.open(os.path.join(self.superpixels_path, str(cam['token']) + '.png')))
            elif self.superpixels_type == 'grounding_sam':
                im_mask_path = os.path.join(self.superpixels_path, cam["filename"]).replace('samples/', '').replace("jpg", "png")
                sp = np.array(Image.open(im_mask_path))
            v_m = (sp != self.ignored_label)
            superpixels.append(sp)  # 超像素

            # Points live in the point sensor frame. So they need to be transformed via
            # global to the image plane.
            # First step: transform the pointcloud to the ego vehicle frame for the
            # timestamp of the sweep.
            cs_record = self.nusc.get(
                "calibrated_sensor", pointsensor["calibrated_sensor_token"]
            )
            pc.rotate(Quaternion(cs_record["rotation"]).rotation_matrix)
            pc.translate(np.array(cs_record["translation"]))

            # Second step: transform from ego to the global frame.
            poserecord = self.nusc.get("ego_pose", pointsensor["ego_pose_token"])
            pc.rotate(Quaternion(poserecord["rotation"]).rotation_matrix)
            pc.translate(np.array(poserecord["translation"]))

            # Third step: transform from global into the ego vehicle frame for the
            # timestamp of the image.
            poserecord = self.nusc.get("ego_pose", cam["ego_pose_token"])
            pc.translate(-np.array(poserecord["translation"]))
            pc.rotate(Quaternion(poserecord["rotation"]).rotation_matrix.T)

            # Fourth step: transform from ego into the camera.
            cs_record = self.nusc.get(
                "calibrated_sensor", cam["calibrated_sensor_token"]
            )
            pc.translate(-np.array(cs_record["translation"]))
            pc.rotate(Quaternion(cs_record["rotation"]).rotation_matrix.T)

            # Fifth step: actually take a "picture" of the point cloud.
            # Grab the depths (camera frame z axis points away from the camera).
            depths = pc.points[2, :]

            # Take the actual picture
            # (matrix multiplication with camera-matrix + renormalization).
            points = view_points(
                pc.points[:3, :],
                np.array(cs_record["camera_intrinsic"]),
                normalize=True,
            )

            # Remove points that are either outside or behind the camera.
            # Also make sure points are at least 1m in front of the camera to avoid
            # seeing the lidar points on the camera
            # casing for non-keyframes which are slightly out of sync.
            points = points[:2].T
            mask = np.ones(depths.shape[0], dtype=bool)  #
            mask = np.logical_and(mask, depths > min_dist)  #
            mask = np.logical_and(mask, points[:, 0] > 0)
            mask = np.logical_and(mask, points[:, 0] < im.shape[1] - 1)  # height
            mask = np.logical_and(mask, points[:, 1] > 0)
            mask = np.logical_and(mask, points[:, 1] < im.shape[0] - 1)  # width
            matching_points = np.where(mask)[0]  # 匹配点的索引
            matching_pixels = np.round(
                np.flip(points[matching_points], axis=1)
            ).astype(np.int64)  # [height, width] -> [width, height]  图像上对应的像素
            p_v_m = v_m[matching_pixels[:, 0], matching_pixels[:, 1]] == True
            dr = np.sum(p_v_m) / p_v_m.shape[0]
            if dr > 0.1:
                matching_points = matching_points[p_v_m]
                matching_pixels = matching_pixels[p_v_m]
            images.append(im / 255)
            # #############################
            # # 保留非0的sp_value 2023/10/4
            # sp_value = sp[matching_pixels[:, 0], matching_pixels[:, 1]]
            # nonzero_mask = ~np.in1d(sp_value, 0)
            # matching_points = matching_points[nonzero_mask]
            # matching_pixels = matching_pixels[nonzero_mask]
            # # inds, counts = np.unique(sp_value, return_counts=True)

            ####
            pairing_points = np.concatenate((pairing_points, matching_points))
            pairing_images = np.concatenate(
                (
                    pairing_images,
                    np.concatenate(
                        (
                            np.ones((matching_pixels.shape[0], 1), dtype=np.int64) * i,  #
                            matching_pixels,
                        ),
                        axis=1,
                    ),
                )
            )

        return pc_ref.T, images, pairing_points, pairing_images, np.stack(superpixels), img_path_list

    def __len__(self):
        # return 120
        return len(self.list_keyframes)

    def __getitem__(self, idx):
        (
            pc,
            images,
            pairing_points,  # 与image相匹配的point
            pairing_images,  # 与pc相匹配的pixel
            superpixels,  # 超像素
            img_path_list
        ) = self.map_pointcloud_to_image(self.list_keyframes[idx])
        superpixels = torch.tensor(superpixels)
        if self.normalize_intensity:
            pc[:, 3] = pc[:, 3] / 255.0

        intensity = torch.tensor(pc[:, 3:])  # (34688) vs (22215)
        pc = torch.tensor(pc[:, :3])
        images = torch.tensor(np.array(images, dtype=np.float32).transpose(0, 3, 1, 2))

        # 点云数据增强
        if self.cloud_transforms:
            pc = self.cloud_transforms(pc)
        if self.mixed_transforms:
            (
                pc,
                intensity,
                images,
                pairing_points,
                pairing_images,
                superpixels,
            ) = self.mixed_transforms(pc, intensity, images, pairing_points, pairing_images, superpixels)

        pc_xyz = pc  # (N,3)
        if self.use_polar:
            pc_xyz = cart2polar(pc_xyz)
        pc_xyz = np.array(pc_xyz)
        intensity = np.array(intensity)
        pc_features = np.concatenate([pc_xyz, intensity], axis=1)

        crop_range = self.max_bound - self.min_bound
        grid_size = np.array(self.grid_size)
        intervals = crop_range / (grid_size - 1)
        point_grid_ind = (np.floor((np.clip(pc_xyz, self.min_bound, self.max_bound) - self.min_bound) / intervals)) \
            .astype(np.int64)

        return (
            pc_features,
            point_grid_ind,  # 每个点的网格index
            images,
            pairing_points,
            pairing_images,
            superpixels
        )


def cart2polar(input_xyz_cart):
    rho = np.sqrt(input_xyz_cart[:, 0] ** 2 + input_xyz_cart[:, 1] ** 2)
    phi = np.arctan2(input_xyz_cart[:, 1], input_xyz_cart[:, 0])
    return np.stack((rho, phi, input_xyz_cart[:, 2]), axis=1)


def polar2cat(input_xyz_polar):
    # print(input_xyz_polar.shape)
    x = input_xyz_polar[0] * np.cos(input_xyz_polar[1])
    y = input_xyz_polar[0] * np.sin(input_xyz_polar[1])
    return np.stack((x, y, input_xyz_polar[2]), axis=0)
