import sys
sys.path.append("..")


import os
import numpy as np

import torch
from torch.utils import data

from nuscenes import NuScenes as NuScenes_devkit

from pyquaternion import Quaternion
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
import copy

class NuScenes(data.Dataset):
    labels_mapping = {
        1: 0,
        5: 0,
        7: 0,
        8: 0,
        10: 0,
        11: 0,
        13: 0,
        19: 0,
        20: 0,
        0: 0,
        29: 0,
        31: 0,
        9: 1,
        14: 2,
        15: 3,
        16: 3,
        17: 4,
        18: 5,
        21: 6,
        2: 7,
        3: 7,
        4: 7,
        6: 7,
        12: 8,
        22: 9,
        23: 10,
        24: 11,
        25: 12,
        26: 13,
        27: 14,
        28: 15,
        30: 16
    }

    CAM_CHANNELS = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                    'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']

    IMAGE_SIZE = (900, 1600)

    def __init__(self, dataroot, split="train", **kwargs):
        self.nusc = NuScenes_devkit(dataroot=dataroot, version='v1.0-trainval', verbose=True)
        self.split = split
        self.ignored_labels = 0
        self.THING_LIST = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.MIN_INST_POINT = 10
        self.STUFF_LIST = [11, 12, 13, 14, 15, 16]
        self.MIN_STUFF_POINT = 50
        self.GRID_SIZE = np.array([300, 400]).reshape([1, 2])

        # self.num_classes = configs['data']['num_classes']

        if self.split == "train":
            select_idx = np.load("./data/nuscenes/nuscenes_train_official.npy")
            self.sample = [self.nusc.sample[i] for i in select_idx]
        elif self.split == "val":
            select_idx = np.load("./data/nuscenes/nuscenes_val_official.npy")
            self.sample = [self.nusc.sample[i] for i in select_idx]
        elif self.split == "test":
            self.sample = self.nusc.sample
        else:
            print("split not implement yet, exit!")
            exit(-1)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.sample)
        # return 12

    def view_lidar_cam_sweeps(self, index):
        sample = self.sample[index]
        lidar_token = sample["data"]["LIDAR_TOP"]
        lidar_channel = self.nusc.get("sample_data", lidar_token)
        # lidar_path = os.path.join(self.nusc.dataroot, lidar_channel["filename"])
        # pts = np.fromfile(lidar_path, dtype=np.float32).reshape([-1, 5])[:, :4]  # N, 4
        lidar_channel_list = [lidar_channel]
        lidar_channel = self.nusc.get('sample_data', lidar_channel['next'])
        while not lidar_channel['is_key_frame']:
            # lidar_path = os.path.join(self.nusc.dataroot, lidar_channel['filename'])
            lidar_channel_list.append(lidar_channel)
            lidar_channel = self.nusc.get('sample_data', lidar_channel['next'])

        cam_channel_list = {}
        for c_name in self.CAM_CHANNELS:
            cam_channel_list[str(c_name)] = []
        for idx, channel in enumerate(self.CAM_CHANNELS):
            cam_token = sample['data'][channel]
            cam_channel = self.nusc.get('sample_data', cam_token)
            cam_channel_list[str(channel)].append(cam_channel)
            cam_channel = self.nusc.get('sample_data', cam_channel['next'])
            while not cam_channel['is_key_frame']:
                cam_channel_list[str(channel)].append(cam_channel)
                cam_channel = self.nusc.get('sample_data', cam_channel['next'])


        feed_dict = {
            'cam_channel_list': cam_channel_list,
            'pcd_list': lidar_channel_list,
        }

        return feed_dict

    def map_lidar_to_image(self, lidar_channel, cam_channel):
        pcl_path = os.path.join(self.nusc.dataroot, lidar_channel["filename"])
        pc_original = LidarPointCloud.from_file(pcl_path)
        pc_ref = pc_original.points

        pc = copy.deepcopy(pc_original)
        # im = np.array(Image.open(os.path.join(self.nusc.dataroot, cam_channel["filename"])))

        # Points live in the point sensor frame. So they need to be transformed via
        # global to the image plane.
        # First step: transform the pointcloud to the ego vehicle frame for the
        # timestamp of the sweep.
        cs_record = self.nusc.get(
            "calibrated_sensor", lidar_channel["calibrated_sensor_token"]
        )
        pc.rotate(Quaternion(cs_record["rotation"]).rotation_matrix)
        pc.translate(np.array(cs_record["translation"]))

        # Second step: transform from ego to the global frame.
        poserecord = self.nusc.get("ego_pose", lidar_channel["ego_pose_token"])
        pc.rotate(Quaternion(poserecord["rotation"]).rotation_matrix)
        pc.translate(np.array(poserecord["translation"]))

        # Third step: transform from global into the ego vehicle frame for the
        # timestamp of the image.
        poserecord = self.nusc.get("ego_pose", cam_channel["ego_pose_token"])
        pc.translate(-np.array(poserecord["translation"]))
        pc.rotate(Quaternion(poserecord["rotation"]).rotation_matrix.T)

        # Fourth step: transform from ego into the camera.
        cs_record = self.nusc.get(
            "calibrated_sensor", cam_channel["calibrated_sensor_token"]
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
        mask = np.logical_and(mask, depths > 1.0)
        mask = np.logical_and(mask, points[:, 0] > 0)
        mask = np.logical_and(mask, points[:, 0] < self.IMAGE_SIZE[1] - 1)
        mask = np.logical_and(mask, points[:, 1] > 0)
        mask = np.logical_and(mask, points[:, 1] < self.IMAGE_SIZE[0] - 1)
        matching_points = np.where(mask)[0]
        # matching_pixels = np.round(
        #     np.flip(points[matching_points], axis=1)
        # ).astype(np.int64)
        matching_pixels = np.round(points[matching_points]).astype(np.int64)

        return matching_points, matching_pixels

    def calc_timestamp_diff(self, index):
        sample = self.sample[index]
        lidar_token = sample["data"]["LIDAR_TOP"]
        lidar_channel = self.nusc.get("sample_data", lidar_token)
        # lidar_path = os.path.join(self.nusc.dataroot, lidar_channel["filename"])
        # pts = np.fromfile(lidar_path, dtype=np.float32).reshape([-1, 5])[:, :4]  # N, 4
        lidar_channel_list = [lidar_channel]
        if lidar_channel['next'] != '':
            lidar_channel = self.nusc.get('sample_data', lidar_channel['next'])
        else:
            lidar_channel = None
        while (lidar_channel is not None) and (not lidar_channel['is_key_frame']):
            # lidar_path = os.path.join(self.nusc.dataroot, lidar_channel['filename'])
            lidar_channel_list.append(lidar_channel)
            lidar_channel = self.nusc.get('sample_data', lidar_channel['next'])

        cam_channel_list = {}
        for c_name in self.CAM_CHANNELS:
            cam_channel_list[str(c_name)] = []
        for idx, channel in enumerate(self.CAM_CHANNELS):
            cam_token = sample['data'][channel]
            cam_channel = self.nusc.get('sample_data', cam_token)
            cam_channel_list[str(channel)].append(cam_channel)
            if cam_channel['next'] != '':
                cam_channel = self.nusc.get('sample_data', cam_channel['next'])
            else:
                cam_channel = None
            while (cam_channel is not None) and (not cam_channel['is_key_frame']):
                cam_channel_list[str(channel)].append(cam_channel)
                cam_channel = self.nusc.get('sample_data', cam_channel['next'])

        pair_list = []
        for cm_list in list(cam_channel_list.values()):
            for c_i, cm_tk in enumerate(cm_list):
                pair = {'cam': cm_tk}
                c_t = cm_tk['timestamp']
                delta = []
                for l_tk in lidar_channel_list:
                    delta.append(c_t - l_tk['timestamp'])
                l_idx = np.argmin(np.fabs(delta))
                pair['lidar'] = lidar_channel_list[l_idx]
                pair['delta_t'] = delta[l_idx]
                pair['l_idx'] = l_idx
                # matching_points, matching_pixels = self.map_lidar_to_image(pair['lidar'], pair['cam'])
                # pair['matching_points'] = matching_points
                # pair['matching_pixels'] = matching_pixels
                pair_list.append(pair)

        feed_dict = {
            'pair_list': pair_list,
        }

        return feed_dict

    def calc_timestamp_diff_1(self, index):
        sample = self.sample[index]
        lidar_token = sample["data"]["LIDAR_TOP"]
        lidar_channel = self.nusc.get("sample_data", lidar_token)
        # lidar_path = os.path.join(self.nusc.dataroot, lidar_channel["filename"])
        # pts = np.fromfile(lidar_path, dtype=np.float32).reshape([-1, 5])[:, :4]  # N, 4
        lidar_channel_list = [lidar_channel]
        if lidar_channel['next'] != '':
            lidar_channel = self.nusc.get('sample_data', lidar_channel['next'])
        else:
            lidar_channel = None
        while (lidar_channel is not None) and (not lidar_channel['is_key_frame']):
            # lidar_path = os.path.join(self.nusc.dataroot, lidar_channel['filename'])
            lidar_channel_list.append(lidar_channel)
            lidar_channel = self.nusc.get('sample_data', lidar_channel['next'])

        cam_channel_list = {}
        for c_name in self.CAM_CHANNELS:
            cam_channel_list[str(c_name)] = []
        for idx, channel in enumerate(self.CAM_CHANNELS):
            cam_token = sample['data'][channel]
            cam_channel = self.nusc.get('sample_data', cam_token)
            cam_channel_list[str(channel)].append(cam_channel)
            if cam_channel['next'] != '':
                cam_channel = self.nusc.get('sample_data', cam_channel['next'])
            else:
                cam_channel = None
            while (cam_channel is not None) and (not cam_channel['is_key_frame']):
                cam_channel_list[str(channel)].append(cam_channel)
                cam_channel = self.nusc.get('sample_data', cam_channel['next'])

        pair_list = []
        for l_tk in lidar_channel_list:
            pair = {'lidar': l_tk,
                    'cam_i': [],
                    'cam': [],
                    'delta_t': [],
                    'sample_idx': index}
            for c_name in self.CAM_CHANNELS:
                delta = []
                for c_i, cm_tk in enumerate(cam_channel_list[str(c_name)]):
                    delta.append(l_tk['timestamp'] - cm_tk['timestamp'])
                c_idx = np.argmin(np.fabs(delta))
                pair['cam_i'].append(c_idx)
                pair['cam'].append(cam_channel_list[str(c_name)][c_idx])
                pair['delta_t'].append(delta[c_idx])
            pair_list.append(pair)

        feed_dict = {
            'pair_list': pair_list,
        }

        return feed_dict

    def save_nkf_pairs(self, index):
        sample = self.sample[index]
        lidar_token = sample["data"]["LIDAR_TOP"]
        lidar_channel = self.nusc.get("sample_data", lidar_token)
        # lidar_path = os.path.join(self.nusc.dataroot, lidar_channel["filename"])
        # pts = np.fromfile(lidar_path, dtype=np.float32).reshape([-1, 5])[:, :4]  # N, 4
        scene = self.nusc.get('scene', sample['scene_token'])
        lidar_channel_list = [lidar_channel]
        if lidar_channel['next'] != '':
            lidar_channel = self.nusc.get('sample_data', lidar_channel['next'])
        else:
            lidar_channel = None
        while (lidar_channel is not None) and (not lidar_channel['is_key_frame']):
            # lidar_path = os.path.join(self.nusc.dataroot, lidar_channel['filename'])
            lidar_channel_list.append(lidar_channel)
            lidar_channel = self.nusc.get('sample_data', lidar_channel['next'])

        cam_channel_list = {}
        for c_name in self.CAM_CHANNELS:
            cam_channel_list[str(c_name)] = []
        for idx, channel in enumerate(self.CAM_CHANNELS):
            cam_token = sample['data'][channel]
            cam_channel = self.nusc.get('sample_data', cam_token)
            cam_channel_list[str(channel)].append(cam_channel)
            if cam_channel['next'] != '':
                cam_channel = self.nusc.get('sample_data', cam_channel['next'])
            else:
                cam_channel = None
            while (cam_channel is not None) and (not cam_channel['is_key_frame']):
                cam_channel_list[str(channel)].append(cam_channel)
                cam_channel = self.nusc.get('sample_data', cam_channel['next'])

        pair_list = []
        for l_tk in lidar_channel_list:
            pair = {'lidar': l_tk,
                    'cam_i': [],
                    'cam': [],
                    'delta_t': [],
                    'sample_idx': index,
                    'sample': sample,
                    'scene': scene}
            for c_name in self.CAM_CHANNELS:
                delta = []
                for c_i, cm_tk in enumerate(cam_channel_list[str(c_name)]):
                    delta.append(l_tk['timestamp'] - cm_tk['timestamp'])
                c_idx = np.argmin(np.fabs(delta))
                pair['cam_i'].append(c_idx)
                pair['cam'].append(cam_channel_list[str(c_name)][c_idx])
                pair['delta_t'].append(delta[c_idx])
            pair_list.append(pair)

        feed_dict = {
            'pair_list': pair_list,
        }

        return feed_dict

    def get_camera_token(self, index):
        sample = self.sample[index]
        camera_tokens = []
        for idx, channel in enumerate(self.CAM_CHANNELS):
            cam_token = sample['data'][channel]
            camera_tokens.append(cam_token)
        return camera_tokens

    def get_lidar_token(self, index):
        sample = self.sample[index]
        scene = self.nusc.get('scene', sample['scene_token'])
        lidar_token = sample["data"]["LIDAR_TOP"]
        return lidar_token

    def __getitem__(self, index):
        sample = self.sample[index]
        lidar_token = sample["data"]["LIDAR_TOP"]
        lidar_channel = self.nusc.get("sample_data", lidar_token)
        # lidar_path = os.path.join(self.nusc.dataroot, lidar_channel["filename"])
        # pts = np.fromfile(lidar_path, dtype=np.float32).reshape([-1, 5])[:, :4]  # N, 4
        lidar_channel_list = [lidar_channel]
        lidar_channel = self.nusc.get('sample_data', lidar_channel['next'])
        while not lidar_channel['is_key_frame']:
            # lidar_path = os.path.join(self.nusc.dataroot, lidar_channel['filename'])
            lidar_channel_list.append(lidar_channel)
            lidar_channel = self.nusc.get('sample_data', lidar_channel['next'])

        cam_channel_list = {}
        for c_name in self.CAM_CHANNELS:
            cam_channel_list[str(c_name)] = []
        for idx, channel in enumerate(self.CAM_CHANNELS):
            cam_token = sample['data'][channel]
            cam_channel = self.nusc.get('sample_data', cam_token)
            cam_channel_list[str(channel)].append(cam_channel)
            cam_channel = self.nusc.get('sample_data', cam_channel['next'])
            while not cam_channel['is_key_frame']:
                cam_channel_list[str(channel)].append(cam_channel)
                cam_channel = self.nusc.get('sample_data', cam_channel['next'])

        pair_list = []
        for cm_list in list(cam_channel_list.values()):
            for c_i, cm_tk in enumerate(cm_list):
                pair = {'cam': cm_tk}
                c_t = cm_tk['timestamp']
                delta = []
                for l_tk in lidar_channel_list:
                    delta.append(c_t - l_tk['timestamp'])
                l_idx = np.argmin(np.fabs(delta))
                pair['lidar'] = lidar_channel_list[l_idx]
                matching_points, matching_pixels = self.map_lidar_to_image(pair['lidar'], pair['cam'])
                pair['matching_points'] = matching_points
                pair['matching_pixels'] = matching_pixels
                pair_list.append(pair)

        feed_dict = {
            'pair_list': pair_list,
        }

        return feed_dict

    @staticmethod
    def collate_fn(batch):
        if isinstance(batch[0], dict):
            ans_dict = {}
            for key in batch[0].keys():
                ans_dict[key] = [sample[key] for sample in batch]
            return ans_dict


def main_calc_delta_t():

    from tqdm import tqdm

    DATA_ROOT = '/home/stf/workspace/datasets/nuscenes'
    dataset = NuScenes(dataroot=DATA_ROOT)
    kf_dt = []
    nkf_dt = []
    for idx in tqdm(range(len(dataset))):
        pair_list = dataset.calc_timestamp_diff(idx)['pair_list']
        for pair in pair_list:
            if pair['cam']['is_key_frame']:
                kf_dt.append(pair['delta_t'])
            else:
                nkf_dt.append(pair['delta_t'])
        if idx !=0 and (idx % 500 == 0):
            print('mean kf_dt:', np.mean(np.fabs(kf_dt)))
            print('std kf_dt:', np.std(np.fabs(kf_dt)))
            print('max kf_dt:', np.max(np.fabs(kf_dt)))
            print('mean nkf_dt:', np.mean(np.fabs(nkf_dt)))
            print('std nkf_dt:', np.std(np.fabs(nkf_dt)))
            print('max nkf_dt:', np.max(np.fabs(nkf_dt)))
    print('mean kf_dt:', np.mean(np.fabs(kf_dt)))
    print('std kf_dt:', np.std(np.fabs(kf_dt)))
    print('max kf_dt:', np.max(np.fabs(kf_dt)))
    print('mean nkf_dt:', np.mean(np.fabs(nkf_dt)))
    print('std nkf_dt:', np.std(np.fabs(nkf_dt)))
    print('max nkf_dt:', np.max(np.fabs(nkf_dt)))


def main_calc_delta_t_1():

    from tqdm import tqdm

    DATA_ROOT = '/home/stf/workspace/datasets/nuscenes'
    dataset = NuScenes(dataroot=DATA_ROOT)
    kf_dt_si = []
    kf_dt = []
    nkf_dt_si = []
    nkf_dt = []
    filtered_nkf = 0
    for idx in tqdm(range(len(dataset))):
        # idx = 27795 # kf_max
        # idx = 5935  # nkf_max
        pair_list = dataset.calc_timestamp_diff_1(idx)['pair_list']
        for pair in pair_list:
            if pair['lidar']['is_key_frame']:
                kf_dt_si.append(pair['sample_idx'])
                kf_dt.append(np.mean(np.fabs(pair['delta_t'])))
            else:
                m_t = np.mean(np.fabs(pair['delta_t']))
                if m_t < 20631 + 4717:
                    nkf_dt_si.append(pair['sample_idx'])
                    nkf_dt.append(np.mean(np.fabs(pair['delta_t'])))
                else:
                    filtered_nkf += 1
        if idx !=0 and (idx % 500 == 0):
            print('mean kf_dt:', np.mean(np.fabs(kf_dt)))
            print('std kf_dt:', np.std(np.fabs(kf_dt)))
            print('max kf_dt:', np.max(np.fabs(kf_dt)))
            print('max_id kf_dt:', kf_dt_si[np.argmax(np.fabs(kf_dt))])
            print('mean nkf_dt:', np.mean(np.fabs(nkf_dt)))
            print('std nkf_dt:', np.std(np.fabs(nkf_dt)))
            print('max nkf_dt:', np.max(np.fabs(nkf_dt)))
            print('max_id nkf_dt:', nkf_dt_si[np.argmax(np.fabs(nkf_dt))])
    print('mean kf_dt:', np.mean(np.fabs(kf_dt)))
    print('std kf_dt:', np.std(np.fabs(kf_dt)))
    print('max kf_dt:', np.max(np.fabs(kf_dt)))
    print('max_id kf_dt:', kf_dt_si[np.argmax(np.fabs(kf_dt))])
    print('mean nkf_dt:', np.mean(np.fabs(nkf_dt)))
    print('std nkf_dt:', np.std(np.fabs(nkf_dt)))
    print('max nkf_dt:', np.max(np.fabs(nkf_dt)))
    print('max_id nkf_dt:', nkf_dt_si[np.argmax(np.fabs(nkf_dt))])
    print('valid nkf:', len(nkf_dt))
    print('filtered nkf:', filtered_nkf)

def main_save_nkf_pairs_new_new():

    from tqdm import tqdm
    import pickle

    # 84
    # DATA_ROOT = '/data/stf/datasets/nuscenes'
    # SWEEPS_PAIR_SAVE_PATH = '/data2/share/new_new_sweeps_pairs_filtered.pkl'
    # 62
    DATA_ROOT = '/data/stf/datasets/nuscenes'
    SWEEPS_PAIR_SAVE_PATH = '/data2/stf/sweeps_pairs_filtered_by_mean.pkl'
    dataset = NuScenes(dataroot=DATA_ROOT)
    kf_dt_si = []
    kf_dt = []
    nkf_dt_si = []
    nkf_dt = []
    filtered_nkf = 0
    saved_pkl = {'keyframe': {}, 'sweeps': {}}
    for idx in tqdm(range(len(dataset))):
        # idx = 27795 # kf_max
        # idx = 5935  # nkf_max
        pair_list = dataset.save_nkf_pairs(idx)['pair_list']
        for pair in pair_list:
            if pair['lidar']['is_key_frame']:
                if saved_pkl['keyframe'].get(pair['scene']['name'], None) is None:
                    saved_pkl['keyframe'][pair['scene']['name']] = []
                    saved_pkl['keyframe'][pair['scene']['name']].append(pair)
                else:
                    saved_pkl['keyframe'][pair['scene']['name']].append(pair)
                kf_dt_si.append(pair['sample_idx'])
                kf_dt.append(np.mean(np.fabs(pair['delta_t'])))
                # saved_pkl['keyframe'].append(pair)
            else:
                # nkf_dt_si.append(pair['sample_idx'])
                # nkf_dt.append(np.mean(np.fabs(pair['delta_t'])))
                # saved_pkl['sweeps'].append(pair)
                m_t = np.mean(np.fabs(pair['delta_t']))
                if m_t < 20631 + 4717:  # 20631 + 4717
                    if saved_pkl['sweeps'].get(pair['scene']['name'], None) is None:
                        saved_pkl['sweeps'][str(pair['scene']['name'])] = []
                    # if saved_pkl['sweeps'][str(pair['scene']['name'])].get(str(pair['sample']['token']), None) is None:
                    #     saved_pkl['sweeps'][str(pair['scene']['name'])][str(pair['sample']['token'])] = []
                    nkf_dt_si.append(pair['sample_idx'])
                    nkf_dt.append(np.mean(np.fabs(pair['delta_t'])))
                    saved_pkl['sweeps'][str(pair['scene']['name'])].append(pair)
                else:
                    filtered_nkf += 1
        if idx !=0 and (idx % 500 == 0):
            print('mean kf_dt:', np.mean(np.fabs(kf_dt)))
            print('std kf_dt:', np.std(np.fabs(kf_dt)))
            print('max kf_dt:', np.max(np.fabs(kf_dt)))
            print('max_id kf_dt:', kf_dt_si[np.argmax(np.fabs(kf_dt))])
            print('mean nkf_dt:', np.mean(np.fabs(nkf_dt)))
            print('std nkf_dt:', np.std(np.fabs(nkf_dt)))
            print('max nkf_dt:', np.max(np.fabs(nkf_dt)))
            print('max_id nkf_dt:', nkf_dt_si[np.argmax(np.fabs(nkf_dt))])
    print('mean kf_dt:', np.mean(np.fabs(kf_dt)))
    print('std kf_dt:', np.std(np.fabs(kf_dt)))
    print('max kf_dt:', np.max(np.fabs(kf_dt)))
    print('max_id kf_dt:', kf_dt_si[np.argmax(np.fabs(kf_dt))])
    print('mean nkf_dt:', np.mean(np.fabs(nkf_dt)))
    print('std nkf_dt:', np.std(np.fabs(nkf_dt)))
    print('max nkf_dt:', np.max(np.fabs(nkf_dt)))
    print('max_id nkf_dt:', nkf_dt_si[np.argmax(np.fabs(nkf_dt))])
    print('valid nkf:', len(nkf_dt))
    print('filtered nkf:', filtered_nkf)
    for k, v in saved_pkl.items():
        print(f'load {len(v)} {k} database infos')
    with open(SWEEPS_PAIR_SAVE_PATH, 'wb') as f:
        pickle.dump(saved_pkl, f)
    print('pkl saved at', SWEEPS_PAIR_SAVE_PATH)


if __name__ == '__main__':
    main_save_nkf_pairs_new_new()