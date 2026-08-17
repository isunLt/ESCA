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


def minkunet_collate_pair_fn_sweeps_find(list_data):
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
        lidar_token,
        kf_flag_list,
        sample_dict_list
    ) = list(zip(*list_data))
    coords = coords[0]
    feats = feats[0]
    images = images[0]
    pairing_points = pairing_points[0]
    pairing_images = pairing_images[0]
    inverse_indexes = inverse_indexes[0]
    superpixels = superpixels[0]
    lidar_token = lidar_token[0]
    kf_flag_list = kf_flag_list[0]
    sample_dict_list = sample_dict_list[0]
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

    return {
        "sinput_C": coords_batch,
        "sinput_F": feats_batch,
        "input_I": images_batch,
        "pairing_points": pairing_points,
        "pairing_images": pairing_images,
        "batch_n_pairings": batch_n_pairings,
        "inverse_indexes": inverse_indexes,
        "superpixels": superpixels_batch,
        'lidar_tokens': lidar_token,
        'keyframe_flag_list': kf_flag_list,
        'sample_dict_list': sample_dict_list
    }

class NuScenesSweepsFindMatchDataset(Dataset):
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
        if phase == 'parametrizing':
            self.list_sweeps = []
            sweeps_pair_list_path = config['sweeps_pair_list_path']
            with open(sweeps_pair_list_path, 'rb') as f:
                sweeps_pair_list = pickle.load(f)
            self.sweeps_pair_list = sweeps_pair_list['sweeps']

        if "cached_nuscenes" in kwargs:
            self.nusc = kwargs["cached_nuscenes"]
        else:
            self.nusc = NuScenes(
                version="v1.0-trainval", dataroot="datasets/nuscenes", verbose=False
            )

        self.list_keyframes = []
        self.list_sweeps = []
        self.list_sample_token = []
        self.list_tmp = []
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
                    if phase == 'parametrizing':
                        # self.create_list_of_scans_with_sweeps(scene)
                        self.create_list_of_scans_with_sweeps_slot(scene)
                    else:
                        self.create_list_of_scans(scene)

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

    def _aggregate_lidar_sweeps(self, sample_ref, nsweeps, scene_name, only_past=False):

        def _agg_sweeps(sweeps_num, scene_name, direction='prev'):
            assert direction in ['prev', 'next']
            current_sd_rec = ref_sd_rec
            sweep_sample = []
            timestamp = []
            t_s = 1
            while len(sweep_sample) < sweeps_num:
                if current_sd_rec[direction] == '':
                    break
                for sw in self.sweeps_pair_list[scene_name]:
                    if sw['lidar']['token'] == current_sd_rec[direction]:
                        sweep_sample.append(
                            {'LIDAR_TOP': sw['lidar']['token'],
                             'CAM_FRONT_LEFT': sw['cam'][0]['token'],
                             'CAM_FRONT': sw['cam'][1]['token'],
                             'CAM_FRONT_RIGHT': sw['cam'][2]['token'],
                             'CAM_BACK_LEFT': sw['cam'][3]['token'],
                             'CAM_BACK': sw['cam'][4]['token'],
                             'CAM_BACK_RIGHT': sw['cam'][5]['token']})
                        timestamp.append(t_s)
                        break
                current_sd_rec = self.nusc.get('sample_data', current_sd_rec[direction])
                t_s += 1
                if current_sd_rec['is_key_frame']:
                    break
            return sweep_sample, timestamp

        # Get reference pose and timestamp.
        ref_sd_token = sample_ref['data']['LIDAR_TOP']
        ref_sd_rec = self.nusc.get('sample_data', ref_sd_token)

        prev_pts, prev_tmp = _agg_sweeps(sweeps_num=nsweeps, scene_name=scene_name, direction='prev')
        prev_tmp = [-x for x in prev_tmp]
        if not only_past:
            # next_pts = _agg_sweeps(sweeps_num=nsweeps, scene_name=scene_name, direction='next')
            next_pts, next_tmp = _agg_sweeps(sweeps_num=(2*nsweeps-len(prev_pts)), scene_name=scene_name, direction='next')
        else:
            next_pts, next_tmp = [], []

        return prev_pts + next_pts, prev_tmp + next_tmp

    def _aggregate_lidar_sweeps_1(self, sample_ref, nsweeps, scene_name):

        def _agg_sweeps(sweeps_num, scene_name, direction='prev'):
            assert direction in ['prev', 'next']
            current_sd_rec = ref_sd_rec
            sweep_sample = []
            while len(sweep_sample) < sweeps_num:
                if current_sd_rec[direction] == '':
                    break
                for sw in self.sweeps_pair_list[scene_name]:
                    if sw['lidar']['token'] == current_sd_rec[direction]:
                        sweep_sample.append(
                            {'LIDAR_TOP': sw['lidar']['token'],
                             'CAM_FRONT_LEFT': sw['cam'][0]['token'],
                             'CAM_FRONT': sw['cam'][1]['token'],
                             'CAM_FRONT_RIGHT': sw['cam'][2]['token'],
                             'CAM_BACK_LEFT': sw['cam'][3]['token'],
                             'CAM_BACK': sw['cam'][4]['token'],
                             'CAM_BACK_RIGHT': sw['cam'][5]['token']})
                        break
                current_sd_rec = self.nusc.get('sample_data', current_sd_rec[direction])
                if current_sd_rec['is_key_frame']:
                    sample_channel = self.nusc.get('sample', current_sd_rec['sample_token'])
                    sweep_sample.append(sample_channel['data'])
                    break
            return sweep_sample

        # Get reference pose and timestamp.
        ref_sd_token = sample_ref['data']['LIDAR_TOP']
        ref_sd_rec = self.nusc.get('sample_data', ref_sd_token)

        next_pts = _agg_sweeps(sweeps_num=nsweeps, scene_name=scene_name, direction='next')

        return next_pts

    def create_list_of_scans_with_sweeps(self, scene):
        # Get first and last keyframe in the scene
        current_sample_token = scene["first_sample_token"]

        # Loop to get all successive keyframes
        list_data = []
        list_sweeps = []
        list_sample_token = []
        list_tmp = []
        while current_sample_token != "":
            current_sample = self.nusc.get("sample", current_sample_token)
            sweeps, tmp = self._aggregate_lidar_sweeps(current_sample, nsweeps=5, scene_name=scene['name'], only_past=False)
            list_data.append(current_sample["data"])
            list_sweeps.append(sweeps)
            list_sample_token.append(current_sample_token)
            list_tmp.append(tmp)
            current_sample_token = current_sample["next"]

        # Add new scans in the list
        self.list_keyframes.extend(list_data)
        self.list_sweeps.extend(list_sweeps)
        self.list_sample_token.extend(list_sample_token)
        self.list_tmp.extend(list_tmp)

    def create_list_of_scans_with_sweeps_slot(self, scene):
        # Get first and last keyframe in the scene
        current_sample_token = scene["first_sample_token"]

        # Loop to get all successive keyframes
        list_data = []
        list_sweeps = []
        list_keyframe_flags = []
        while current_sample_token != "":
            current_sample = self.nusc.get("sample", current_sample_token)
            sweeps = self._aggregate_lidar_sweeps_1(current_sample, nsweeps=10, scene_name=scene['name'])
            list_data.append(current_sample["data"])
            list_sweeps.append(sweeps)
            current_sample_token = current_sample["next"]

        # Add new scans in the list
        self.list_keyframes.extend(list_data)
        self.list_sweeps.extend(list_sweeps)


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
        camera_list = [
            "CAM_FRONT_LEFT",
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK_LEFT",
            "CAM_BACK",
            "CAM_BACK_RIGHT",
        ]
        # camera_list = []
        # for x in list(data.keys()):
        #     if 'CAM' in x:
        #         camera_list.append(x)
        # if self.shuffle:
        #     np.random.shuffle(camera_list)
        for i, camera_name in enumerate(camera_list):
            # if data.get(camera_name, None) is None:
            #     continue
            pc = copy.deepcopy(pc_original)
            cam = self.nusc.get("sample_data", data[camera_name])
            im = np.array(Image.open(os.path.join(self.nusc.dataroot, cam["filename"])))
            # im = np.zeros(shape=(900, 1600, 3), dtype=np.uint8)
            if pointsensor['is_key_frame']:
                sp_path = os.path.join(self.superpixels_path, 'keyframes', str(cam['token']) + '.png')
            else:
                sp_path = os.path.join(self.superpixels_path, 'sweeps', str(cam['token']) + '.png')

            sp = np.array(Image.open(sp_path))
            v_m = (sp != 0)
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
            # dr = np.sum(p_v_m) / p_v_m.shape[0]
            # if dr < 0.1:
            #     print('Warning: %s drop %f of the points' % (data["LIDAR_TOP"], np.round((1.0-dr) * 100, 2)))
            # matching_points = matching_points[p_v_m]
            # matching_pixels = matching_pixels[p_v_m]

            # tmp = sp[matching_pixels[:, 0], matching_pixels[:, 1]]
            # min_tmp = np.min(tmp)
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

    def load_sample_from_dict(self, sample):
        (
            pc,
            images,
            pairing_points,
            pairing_images,
            superpixels,
        ) = self.map_pointcloud_to_image(sample)
        # print('Point Loaded, idx=%d, token=%s' % (idx, str(self.list_keyframes[idx]['LIDAR_TOP'])))
        # print('idx:', idx)
        # print('lidar_token:', self.list_keyframes[idx]['LIDAR_TOP'])
        superpixels = torch.tensor(superpixels)

        intensity = torch.tensor(pc[:, 3:])
        pc = torch.tensor(pc[:, :3])
        images = torch.tensor(np.array(images, dtype=np.float32).transpose(0, 3, 1, 2))  # B,C,H,W

        if self.cloud_transforms:
            pc = self.cloud_transforms(pc)
        # print('Point Aug')
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
        # print('Mix Aug')
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

        return (
            discrete_coords,
            unique_feats,
            images,
            pairing_points,
            pairing_images,
            inverse_indexes,
            superpixels,
            str(sample['LIDAR_TOP']),
        )

    def __getitem__(self, idx):
        discrete_coords_list = []
        unique_feats_list = []
        images_list = []
        pairing_points_list = []
        pairing_images_list = []
        inverse_indexes_list = []
        superpixels_list = []
        lidar_token_list = []
        sample_dict_list = []
        keyframe_flag_list = []
        (
            discrete_coords,
            unique_feats,
            images,
            pairing_points,
            pairing_images,
            inverse_indexes,
            superpixels,
            lidar_token,
        ) = self.load_sample_from_dict(self.list_keyframes[idx])
        discrete_coords_list.append(discrete_coords)
        unique_feats_list.append(unique_feats)
        images_list.append(images)
        pairing_points_list.append(pairing_points)
        pairing_images_list.append(pairing_images)
        inverse_indexes_list.append(inverse_indexes)
        superpixels_list.append(superpixels)
        lidar_token_list.append(lidar_token)
        keyframe_flag_list.append(len(self.list_keyframes[idx]) > 7)
        sample_dict_list.append(self.list_keyframes[idx])

        for lidar_sample in self.list_sweeps[idx]:
            (
                discrete_coords,
                unique_feats,
                images,
                pairing_points,
                pairing_images,
                inverse_indexes,
                superpixels,
                lidar_token,
            ) = self.load_sample_from_dict(lidar_sample)
            discrete_coords_list.append(discrete_coords)
            unique_feats_list.append(unique_feats)
            images_list.append(images)
            pairing_points_list.append(pairing_points)
            pairing_images_list.append(pairing_images)
            inverse_indexes_list.append(inverse_indexes)
            superpixels_list.append(superpixels)
            lidar_token_list.append(lidar_token)
            keyframe_flag_list.append(len(lidar_sample) > 7)
            sample_dict_list.append(lidar_sample)

        return (
            discrete_coords_list,
            unique_feats_list,
            images_list,
            pairing_points_list,
            pairing_images_list,
            inverse_indexes_list,
            superpixels_list,
            lidar_token_list,
            keyframe_flag_list,
            sample_dict_list
        )


def make_data_loader(config, phase, num_threads=0):
    # instantiate the dataset
    assert config['batch_size'] == 1, "Batch size must be 1 for sweeps find dataset"
    assert config['num_gpus'] == 1, "Number of GPUs must be 1 for sweeps find dataset"
    dset = NuScenesSweepsFindMatchDataset(phase=phase, config=config)
    collate_fn = minkunet_collate_pair_fn_sweeps_find
    batch_size = config["batch_size"] // config["num_gpus"]

    # create the loader
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=phase == "parametrizing",
        num_workers=num_threads,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=phase == "train",
        worker_init_fn=lambda id: np.random.seed(torch.initial_seed() // 2 ** 32 + id),
    )
    return loader

