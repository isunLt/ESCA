import os
import copy
import torch
import numpy as np
from PIL import Image
import MinkowskiEngine as ME
from pyquaternion import Quaternion
from torch.utils.data import Dataset
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import LidarPointCloud
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


def minkunet_collate_pair_fn_nusc_semkitti(list_data):

    def _minkunet_collate_pair_fn(list_data):
        """
        Collate function adapted for creating batches with MinkowskiEngine.
        """
        (
            coords,
            feats,
            images,
            pairing_points,
            pairing_images,
            inverse_indexes,
            superpixels,
            # lidar_token,
        ) = list(zip(*list_data))
        batch_n_points, batch_n_pairings = [], []

        offset = 0
        offset_im = 0
        for batch_id in range(len(coords)):

            # Move batchids to the beginning
            coords[batch_id][:, 0] = batch_id
            pairing_points[batch_id][:] += offset
            # pairing_images[batch_id][:, 0] += batch_id * images[0].shape[0]
            pairing_images[batch_id][:, 0] += offset_im

            batch_n_points.append(coords[batch_id].shape[0])
            batch_n_pairings.append(pairing_points[batch_id].shape[0])
            offset += coords[batch_id].shape[0]
            offset_im += images[batch_id].shape[0]

        # Concatenate all lists
        coords_batch = torch.cat(coords, 0).int()
        pairing_points = torch.tensor(np.concatenate(pairing_points))
        pairing_images = torch.tensor(np.concatenate(pairing_images))
        feats_batch = torch.cat(feats, 0).float()
        images_batch = torch.cat(images, 0).float()
        superpixels_batch = torch.tensor(np.concatenate(superpixels))
        # cluster_mask_batch = torch.cat(cluster_mask, 0).long()

        return {
            "sinput_C": coords_batch,
            "sinput_F": feats_batch,
            "input_I": images_batch,
            "pairing_points": pairing_points,
            "pairing_images": pairing_images,
            "batch_n_pairings": batch_n_pairings,
            "inverse_indexes": inverse_indexes,
            "superpixels": superpixels_batch,
            # 'lidar_tokens': lidar_token,
        }


    ret_dict = {}
    nusc_list = [x['nusc'] for x in list_data]
    ret_dict['nusc'] = _minkunet_collate_pair_fn(nusc_list)
    kitti_list = [x['kitti'] for x in list_data]
    ret_dict['kitti'] = _minkunet_collate_pair_fn(kitti_list)

    return ret_dict

KITTI_TRAIN_SET = {0, 1, 2, 3, 4, 5, 6, 7, 9, 10}
KITTI_VALIDATION_SET = {8}
KITTI_TEST_SET = {11, 12, 13, 14, 15, 16, 17, 18, 19, 20}

class NuScenesSemKITTIMatchDataset(Dataset):
    """
    Dataset matching a 3D points cloud and an image using projection.
    """

    def __init__(
        self,
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
        self.voxel_size = config["voxel_size"]
        self.cylinder = config["cylindrical_coordinates"]
        self.superpixels_type = config["superpixels_type"]
        self.bilinear_decoder = config["decoder"] == "bilinear"
        self.superpixels_path = config['superpixels_path']
        self.superpixels_path_kitti = config['superpixels_path_kitti']
        self.ignored_label = config['ignored_label']

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
        if phase in ("train", "val", "test"):
            phase_scenes = create_splits_scenes()[phase]
        elif phase == "parametrizing":
            phase_scenes = list(
                set(create_splits_scenes()["train"]) - set(CUSTOM_SPLIT)
            )
        elif phase == "verifying":
            phase_scenes = CUSTOM_SPLIT
        # create a list of camera & lidar scans
        for scene_idx in range(len(self.nusc.scene)):
            scene = self.nusc.scene[scene_idx]
            if scene["name"] in phase_scenes:
                skip_counter += 1
                if skip_counter % skip_ratio == 0:
                    self.create_list_of_scans(scene)

        self.list_keyframes_semkitti = []

        if phase in ("train", "parametrizing"):
            kitti_phase_set = KITTI_TRAIN_SET
        elif phase in ("val", "verifying"):
            kitti_phase_set = KITTI_VALIDATION_SET
        elif phase == "test":
            kitti_phase_set = KITTI_TEST_SET

        for num in kitti_phase_set:
            directory = next(
                os.walk(
                    f"datasets/semantickitti/sequences/{num:0>2d}/velodyne"
                )
            )
            self.list_keyframes_semkitti.extend(
                map(
                    lambda x: f"datasets/semantickitti/sequences/"
                    f"{num:0>2d}/velodyne/" + x,
                    directory[2],
                )
            )
        self.list_keyframes_semkitti = sorted(self.list_keyframes_semkitti)[::skip_ratio]
        if len(self.list_keyframes_semkitti) < len(self.list_keyframes):
            d = len(self.list_keyframes) - len(self.list_keyframes_semkitti)
            self.list_keyframes_semkitti += self.list_keyframes_semkitti[:d]

        assert len(self.list_keyframes_semkitti) >= len(self.list_keyframes)

        self.P_dict = {}
        self.Tr_dict = {}
        for seq in kitti_phase_set:
            with open(os.path.join('datasets/semantickitti/sequences', f"{seq:0>2d}", 'calib.txt'), 'r') as calib:
                P = []
                for idx in range(4):
                    line = calib.readline().rstrip('\n')[4:]
                    data = line.split(" ")
                    P.append(np.array(data, dtype=np.float32).reshape(3, -1))
                self.P_dict[seq] = P[2]
                line = calib.readline().rstrip('\n')[4:]
                data = line.split(" ")
                self.Tr_dict[seq] = np.array(data, dtype=np.float32).reshape((3, -1))


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
        pc_ref = pc_original.points

        images = []
        superpixels = []
        pairing_points = np.empty(0, dtype=np.int64)
        pairing_images = np.empty((0, 3), dtype=np.int64)
        # camera_list = [
        #     "CAM_FRONT",
        #     "CAM_FRONT_RIGHT",
        #     "CAM_BACK_RIGHT",
        #     "CAM_BACK",
        #     "CAM_BACK_LEFT",
        #     "CAM_FRONT_LEFT",
        # ]
        camera_list = []
        for x in list(data.keys()):
            if 'CAM' in x:
                camera_list.append(x)
        if self.shuffle:
            np.random.shuffle(camera_list)
        for i, camera_name in enumerate(camera_list):
            # if data.get(camera_name, None) is None:
            #     continue
            pc = copy.deepcopy(pc_original)
            cam = self.nusc.get("sample_data", data[camera_name])
            im = np.array(Image.open(os.path.join(self.nusc.dataroot, cam["filename"])))
            if pointsensor['is_key_frame']:
                sp_path = os.path.join(self.superpixels_path, 'keyframes', str(cam['token']) + '_dino_mask.bin')
            else:
                sp_path = os.path.join(self.superpixels_path, 'sweeps', str(cam['token']) + '_dino_mask.bin')

            sp = np.fromfile(sp_path, dtype=np.uint8).reshape(900, 1600).astype(np.int32)
            v_m = (sp != self.ignored_label)
            superpixels.append(sp)

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
            mask = np.ones(depths.shape[0], dtype=bool)
            mask = np.logical_and(mask, depths > min_dist)
            mask = np.logical_and(mask, points[:, 0] > 0)
            mask = np.logical_and(mask, points[:, 0] < im.shape[1] - 1)
            mask = np.logical_and(mask, points[:, 1] > 0)
            mask = np.logical_and(mask, points[:, 1] < im.shape[0] - 1)
            matching_points = np.where(mask)[0]
            matching_pixels = np.round(
                np.flip(points[matching_points], axis=1)
            ).astype(np.int64)
            p_v_m = v_m[matching_pixels[:, 0], matching_pixels[:, 1]] == True
            dr = np.sum(p_v_m) / p_v_m.shape[0]
            if dr > 0.1:
                matching_points = matching_points[p_v_m]
                matching_pixels = matching_pixels[p_v_m]
            images.append(im / 255)
            pairing_points = np.concatenate((pairing_points, matching_points))  # 视野内点的序号
            pairing_images = np.concatenate(
                (
                    pairing_images,
                    np.concatenate(
                        (
                            np.ones((matching_pixels.shape[0], 1), dtype=np.int64) * i,  # 每个点对应的像素坐标 [图像id, h, w]
                            matching_pixels,
                        ),
                        axis=1,
                    ),
                )
            )
        return pc_ref.T, images, pairing_points, pairing_images, np.stack(superpixels)

    def __len__(self):
        return len(self.list_keyframes)

    def _get_sample_by_dict(self, token_dict: dict):
        (
            pc,
            images,
            pairing_points,
            pairing_images,
            superpixels,
        ) = self.map_pointcloud_to_image(token_dict)
        # print('Point Loaded, idx=%d, token=%s' % (idx, str(self.list_keyframes[idx]['LIDAR_TOP'])))
        # print('idx:', idx)
        # print('lidar_token:', self.list_keyframes[idx]['LIDAR_TOP'])
        superpixels = torch.tensor(superpixels)

        intensity = torch.tensor(pc[:, 3:])
        pc = torch.tensor(pc[:, :3])
        images = torch.tensor(np.array(images, dtype=np.float32).transpose(0, 3, 1, 2))  # B,C,H,W

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
            ) = self.mixed_transforms(
                pc, intensity, images, pairing_points, pairing_images, superpixels
            )
        if self.cylinder:
            # Transform to cylinder coordinate and scale for voxel size
            x, y, z = pc.T
            rho = torch.sqrt(x ** 2 + y ** 2) / self.voxel_size
            phi = torch.atan2(y, x) * 180 / np.pi  # corresponds to a split each 1°
            z = z / self.voxel_size
            coords_aug = torch.cat((rho[:, None], phi[:, None], z[:, None]), 1)
        else:
            coords_aug = pc / self.voxel_size

        # Voxelization with MinkowskiEngine
        discrete_coords, indexes, inverse_indexes = ME.utils.sparse_quantize(
            coords_aug.contiguous(), return_index=True, return_inverse=True
        )
        # print('Voxelize')
        # indexes here are the indexes of points kept after the voxelization
        pairing_points = inverse_indexes[pairing_points]

        unique_feats = intensity[indexes]

        discrete_coords = torch.cat(
            (
                torch.zeros(discrete_coords.shape[0], 1, dtype=torch.int32),
                discrete_coords,
            ),
            1,
        )
        # print('Dataset End')

        return (
            discrete_coords,
            unique_feats,
            images,
            pairing_points,
            pairing_images,
            inverse_indexes,
            superpixels,
            # token_dict['LIDAR_TOP'],
        )


    def _mappcd2img_kitti(self, index, pts, sp, im_size):

        pairing_points = np.empty(0, dtype=np.int64)
        pairing_images = np.empty((0, 3), dtype=np.int64)

        seq = int(self.list_keyframes_semkitti[index].split('/')[-3])
        P, Tr = self.P_dict[seq], self.Tr_dict[seq]
        mask = np.ones(pts.shape[0], dtype=bool)
        mask = np.logical_and(mask, pts[:, 0] > 0.0)
        pts_homo = np.column_stack((pts, np.array([1] * pts.shape[0], dtype=pts.dtype)))
        Tr_homo = np.row_stack((Tr, np.array([0, 0, 0, 1], dtype=Tr.dtype)))
        pixel_coord = np.matmul(Tr_homo, pts_homo.T)
        pixel_coord = np.matmul(P, pixel_coord).T

        pixel_coord = pixel_coord / (pixel_coord[:, 2].reshape(-1, 1))
        pixel_coord = pixel_coord[:, :2]

        mask = np.logical_and(mask, pixel_coord[:, 0] > 0)
        mask = np.logical_and(mask, pixel_coord[:, 0] < im_size[1] - 1)
        mask = np.logical_and(mask, pixel_coord[:, 1] > 0)
        mask = np.logical_and(mask, pixel_coord[:, 1] < im_size[0] - 1)
        matching_points = np.where(mask)[0]
        matching_pixels = np.round(
            np.flip(pixel_coord[matching_points], axis=1)
        ).astype(np.int64)

        v_m = (sp != self.ignored_label)
        p_v_m = v_m[matching_pixels[:, 0], matching_pixels[:, 1]] == True
        dr = np.sum(p_v_m) / p_v_m.shape[0]
        if dr > 0.1:
            matching_points = matching_points[p_v_m]
            matching_pixels = matching_pixels[p_v_m]

        pairing_points = np.concatenate((pairing_points, matching_points))  # 视野内点的序号
        pairing_images = np.concatenate(
            (
                pairing_images,
                np.concatenate(
                    (
                        np.ones((matching_pixels.shape[0], 1), dtype=np.int64) * 0,  # 每个点对应的像素坐标 [图像id, h, w]
                        matching_pixels,
                    ),
                    axis=1,
                ),
            )
        )
        return pairing_points, pairing_images

    def _get_kitti_sample_by_index(self, index):
        lidar_file = self.list_keyframes_semkitti[index]
        pc = np.fromfile(lidar_file, dtype=np.float32).reshape((-1, 4))
        im_path = lidar_file.replace('velodyne', 'image_2')[:-3] + 'png'
        images = np.array(Image.open(im_path), dtype=np.float32) / 255.0
        sp_path = im_path.replace('image_2', 'image_2_dino_mask').split('/')[2:]
        sp_path = os.path.join(self.superpixels_path_kitti, sp_path[0], sp_path[1], sp_path[2], sp_path[3])
        superpixels = np.array(Image.open(sp_path))
        pairing_points, pairing_images = self._mappcd2img_kitti(index, pts=pc[:, :3], sp=superpixels, im_size=images.shape)

        superpixels = torch.tensor(np.expand_dims(superpixels, axis=0))

        intensity = torch.tensor(pc[:, 3:])
        pc = torch.tensor(pc[:, :3])
        images = torch.tensor(np.expand_dims(images, axis=0).transpose(0, 3, 1, 2))  # B,C,H,W

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
            ) = self.mixed_transforms(
                pc, intensity, images, pairing_points, pairing_images, superpixels=superpixels
            )

        if self.cylinder:
            # Transform to cylinder coordinate and scale for voxel size
            x, y, z = pc.T
            rho = torch.sqrt(x ** 2 + y ** 2) / self.voxel_size
            phi = torch.atan2(y, x) * 180 / np.pi  # corresponds to a split each 1°
            z = z / self.voxel_size
            coords_aug = torch.cat((rho[:, None], phi[:, None], z[:, None]), 1)
        else:
            coords_aug = pc / self.voxel_size

        # Voxelization with MinkowskiEngine
        discrete_coords, indexes, inverse_indexes = ME.utils.sparse_quantize(
            coords_aug.contiguous(), return_index=True, return_inverse=True
        )
        # indexes here are the indexes of points kept after the voxelization
        pairing_points = inverse_indexes[pairing_points]

        unique_feats = intensity[indexes]

        discrete_coords = torch.cat(
            (
                torch.zeros(discrete_coords.shape[0], 1, dtype=torch.int32),
                discrete_coords,
            ),
            1,
        )

        return (
            discrete_coords,
            unique_feats,
            images,
            pairing_points,
            pairing_images,
            inverse_indexes,
            superpixels,
            # str(self.list_keyframes[idx]['LIDAR_TOP'])
        )


    def __getitem__(self, idx):
        ret_dict = {}
        kf_tokens = self.list_keyframes[idx]
        kf_ret = self._get_sample_by_dict(kf_tokens)
        ret_dict['nusc'] = kf_ret
        ret_dict['kitti'] = self._get_kitti_sample_by_index(idx)
        return ret_dict



