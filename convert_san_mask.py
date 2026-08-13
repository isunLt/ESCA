import os

import PIL
import numpy as np
from PIL import Image
import json
from glob import glob
from tqdm import tqdm
import torch

class CategoryProcessor:

    def __init__(self):

        self.categories = [
            ('car', 'sedan', 'hatch-back', 'wagon', 'van', 'mini-van', 'SUV', 'jeep'),
            ('truck', 'pickup truck', 'lorry truck', 'semi truck', 'personal use truck', 'cargo hauling truck',
             'trailer truck'),
            ('bendy bus', 'articulated bus', 'multi-section shuttle'),
            ('bus', 'standard bus', 'city bus', 'rigid bus'),
            ('construction vehicle', 'crane', 'bulldozer', 'dump truck'),
            ('motorcycle', 'a person on a motorcycle', 'scooter', 'vespa'),
            ('bicycle', 'road bicycle', 'mountain bike', 'electric bike', 'a person on a bicycle'),
            ('bicycle_rack', 'bicycle parking', 'bike rack', 'cycle stand'),
            ('vehicle trailer', 'truck trailer', 'car trailer', 'motorcycle trailer', 'container on a truck'),
            ('police vehicle', 'police car', 'police motorcycle', 'police bicycle'),
            ('ambulance', 'emergency ambulance', 'medical transport'),
            ('human', 'person', 'people', 'adult', 'walking adult', 'mannequin'),
            ('walking child', 'child'),
            ('construction worker', 'construction site worker'),
            ('stroller', 'baby stroller', 'child stroller', 'a stroller with a child'),
            ('wheelchair', 'manual wheelchair', 'a person on a manual wheelchair', 'electric wheelchair',
             'a person on a electric wheelchair'),
            ('skateboard', 'a person on a skateboard', 'segway', 'a person on a segway', 'scooter',
             'a person on a scooter'),
            ('police_officer', 'traffic police', 'patrolling officer'),
            ('animal', 'cat', 'dog', 'deer', 'bird'),
            ('cone', 'safety cone'),
            ('barrier', 'construction zone barrier', 'traffic barrier'),
            ('dolley', 'wheelbarrow', 'shopping cart', 'garbage-bin with wheels'),
            ('obstacle', 'full trash bag', 'construction material'),
            ('driveable surface', 'highway', 'cement road', 'asphalt road', 'road', 'gravel road', 'paved road',
             'unpaved road'),
            ('walkway', 'sidewalk', 'bike path', 'traffic island'),
            ('grass', 'soil', 'sand', ' rolling hills', 'earth', 'ground level horizontal vegetation (< 20 cm tall)'),
            ('rail track', 'stairs with at most 3 steps', 'lake', 'river'),
            ('man-made object', 'building', 'house', 'premises', 'structure', 'part of construction', 'windowspane',
             'door', 'brick wall', 'wall', 'ceiling', 'guard rail', 'fence', 'pole', 'drainage', 'hydrant', 'flag',
             'street sign', 'electric circuit box', 'parking meter', 'stairs with more than 3 steps', 'utility pole',
             'signboard', 'road sign', 'traffic light', 'bus shelter', 'trash bin without wheels', 'fire hydrant',
             'industrial building', 'chair', 'ladder', 'pavilion', 'plate', 'billboard', 'street lamp', 'bench',
             'one side of the building', 'bridge', 'lamp', 'banner', 'generater', 'telephone booth', 'cell box',
             'pavilion', 'tower', 'curtain', 'screen'),
            ('bush', 'tree', 'potted plant', 'plant'),
            ('undistinguishable object', 'sky', 'cloudy sky', 'clear sky', 'overcast'),
            ('bonnet under the picture',)

        ]
        self.contiguous_id_to_class_id_map = []
        for idx, category in enumerate(self.categories):
            self.contiguous_id_to_class_id_map += [idx] * len(category)
        self.contiguous_id_to_class_id_map = np.array(self.contiguous_id_to_class_id_map, dtype=np.uint8)
        self.class_id_to_nusc_class_id = {
            0: 17, 1: 23, 2: 15, 3: 16, 4: 18, 5: 21, 6: 14, 7: 13, 8: 22, 9: 20, 10: 19,
            11: 2, 12: 3, 13: 4, 14: 7, 15: 8, 16: 5, 17: 6, 18: 1, 19: 12, 20: 9,
            21: 11, 22: 10, 23: 24, 24: 26, 25: 27, 26: 25, 27: 28, 28: 30, 29: 29, 30: 31,
        }
        self.class_id_to_nusc_class_id = np.array(list(self.class_id_to_nusc_class_id.values()), dtype=np.uint8)
        self.nusc_class_id_to_nusc_semantic_id = {
            0: 0, 1: 0, 2: 7, 3: 7, 4: 7, 5: 0, 6: 7, 7: 0, 8: 0, 9: 1, 10: 0, 11: 0,
            12: 8, 13: 0, 14: 2, 15: 3, 16: 3, 17: 4, 18: 5, 19: 0, 20: 0, 21: 6, 22: 9,
            23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 29: 0, 30: 16, 31: 0,
        }

    def get_stuff_classes_dic(self):
        results = []
        for category in self.categories:
            category_list = [elem for elem in category]
            results += category_list
        return results

    def map_contiguous_id_to_class_id(self, idx_list):
        if isinstance(idx_list, list):
            tmp = np.array(idx_list, dtype=np.uint8)
        elif isinstance(idx_list, np.ndarray):
            tmp = idx_list
        elif isinstance(idx_list, torch.Tensor):
            tmp = idx_list.detach.cpu().numpy()
        return self.contiguous_id_to_class_id_map[tmp]

    def map_class_id_to_nusc_class_id(self, idx_list):
        if isinstance(idx_list, list):
            tmp = np.array(idx_list, dtype=np.uint8)
        elif isinstance(idx_list, np.ndarray):
            tmp = idx_list
        elif isinstance(idx_list, torch.Tensor):
            tmp = idx_list.detach.cpu().numpy()
        return self.class_id_to_nusc_class_id[tmp]

    def map_nusc_class_id_to_nusc_semantic_id(self, idx_list):
        if isinstance(idx_list, list):
            tmp = np.array(idx_list, dtype=np.uint8)
        elif isinstance(idx_list, np.ndarray):
            tmp = idx_list
        elif isinstance(idx_list, torch.Tensor):
            tmp = idx_list.detach.cpu().numpy()
        return np.vectorize(self.nusc_class_id_to_nusc_semantic_id.__getitem__)(tmp)

def convert_san_mask():
    nusc_categories_processor = CategoryProcessor()
    src_path = '/data2/stf/superpixels_SAN'
    dst_path = '/data2/stf/superpixels_semantic_SAN'
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)
    file_list = glob(os.path.join(src_path, '*.png'))
    for file_name in tqdm(file_list):
        vfm_mask_path = file_name
        vfm_mask = np.array(Image.open(vfm_mask_path)) - 1
        with open(vfm_mask_path.replace('.png', '.json'), 'r') as file:
            sinfo = json.load(file)
            file.close()
        category_ids = np.array([item["category_id"] for item in sinfo]).reshape((-1, 1))
        vfm_mask = category_ids[vfm_mask]
        vfm_mask = nusc_categories_processor.map_contiguous_id_to_class_id(vfm_mask)
        vfm_mask = nusc_categories_processor.map_class_id_to_nusc_class_id(vfm_mask)
        vfm_mask = nusc_categories_processor.map_nusc_class_id_to_nusc_semantic_id(vfm_mask)
        img_path = os.path.join(dst_path, os.path.basename(file_name))
        h, w, _ = vfm_mask.shape
        im = Image.fromarray(vfm_mask.reshape(h, w).astype(np.uint8))
        im.save(img_path)


def main_convert_san_mask_multiprocess():

    def process_one_sequence(file_name: list):
        vfm_mask_path = file_name
        vfm_mask = np.array(Image.open(vfm_mask_path)) - 1
        with open(vfm_mask_path.replace('.png', '.json'), 'r') as file:
            sinfo = json.load(file)
            file.close()
        category_ids = np.array([item["category_id"] for item in sinfo]).reshape((-1, 1))
        vfm_mask = category_ids[vfm_mask]
        vfm_mask = nusc_categories_processor.map_contiguous_id_to_class_id(vfm_mask)
        vfm_mask = nusc_categories_processor.map_class_id_to_nusc_class_id(vfm_mask)
        vfm_mask = nusc_categories_processor.map_nusc_class_id_to_nusc_semantic_id(vfm_mask)
        img_path = os.path.join(dst_path, os.path.basename(file_name))
        h, w, _ = vfm_mask.shape
        im = Image.fromarray(vfm_mask.reshape(h, w).astype(np.uint8))
        im.save(img_path)

    import mlcrate as mlc
    nusc_categories_processor = CategoryProcessor()
    # src_path = '/data2/stf/superpixels_SAN'
    src_path = '/nvme0/chm/superpixels/nuscenes/superpixels_SAN'
    # dst_path = '/data2/stf/superpixels_semantic_SAN'
    dst_path = '/nvme0/stf/superpixels_semantic_SAN'
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)
    file_list = glob(os.path.join(src_path, '*.png'))
    pool = mlc.SuperPool(32)
    pool.map(process_one_sequence, file_list, description='convert san mask')


if __name__ == '__main__':
    # convert_san_mask()
    main_convert_san_mask_multiprocess()