import os.path

import torch
from tqdm import tqdm
from copy import deepcopy
from MinkowskiEngine import SparseTensor
from utils.metrics import compute_IoU
import pickle


CLASSES_NUSCENES = [
    "barrier",
    "bicycle",
    "bus",
    "car",
    "construction_vehicle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "trailer",
    "truck",
    "driveable_surface",
    "other_flat",
    "sidewalk",
    "terrain",
    "manmade",
    "vegetation",
]

CLASSES_KITTI = [
    "car",
    "bicycle",
    "motorcycle",
    "truck",
    "other-vehicle",
    "person",
    "bicyclist",
    "motorcyclist",
    "road",
    "parking",
    "sidewalk",
    "other-ground",
    "building",
    "fence",
    "vegetation",
    "trunk",
    "terrain",
    "pole",
    "traffic-sign",
]

CLASSES_WAYMO = [
    'car',  #
    'truck',  #
    'bus',  #
    'other_vehicle',  #
    'motorcyclist',  #
    'bicyclist',  #
    'pedestrian',  #
    'sign',  #
    'traffic_light',  #
    'pole',  #
    'construction_cone',  #
    'bicycle',  #
    'motorcycle',  #
    'building',  #
    'vegetation',  #
    'tree_trunk',  #
    'curb',  # 路沿
    'road',  #
    'lane_marker',  #
    'other_ground',  #
    'walkable',  #
    'sidewalk'  #
]

# CLASSES_SYNTH4D = [
#     "vehicle",
#     "pedestrian",
#     "road",
#     "sidewalk",
#     "terrain",
#     "building",
#     "vegetation"
# ]

CLASSES_SYNTH4D = [
    'building',
    'fences',
    'other',
    'pedestrian',
    'pole',
    'road_lines',
    'road',
    'sidewalk',
    'vegetation',
    'vehicle',
    'wall',
    'traffic_sign',
    'sky',
    'ground',
    'bridge',
    'rail_track',
    'guardrail',
    'traffic_light',
    'static',
    'dynamic',
    'water',
    'terrain'
]

CLASSES_SEMPOSS = [
    "people",
    "rider",
    "car",
    "truck",
    "plants",
    "traffic-sign",
    "pole",
    "trashcan",
    "building",
    "cone/stone",
    "fence",
    "bike",
    "ground",
]

CLASSES_SEMSTF = [
    "car",
    "bicycle",
    "motorcycle",
    "truck",
    "other-vehicle",
    "person",
    "bicyclist",
    "motorcyclist",
    "road",
    "parking",
    "sidewalk",
    "other-ground",
    "building",
    "fence",
    "vegetation",
    "trunk",
    "terrain",
    "pole",
    "traffic-sign",
]

CLASSES_RELLIS3D = [
    "grass",
    "tree",
    "pole",
    "water",
    "vehicle",
    "log",
    "person",
    "fence",
    "bush",
    "concrete",
    "barrier",
    "puddle",
    "mud",
    "rubble",
]

CLASSES_SYNLIDAR = [
    "car",
    "pick-up",
    "truck",
    "bus",
    "bicycle",
    "motorcycle",
    "other-vehicle",
    "road",
    "sidewalk",
    "parking",
    "other-ground",
    "female",
    "male",
    "kid",
    "group",
    "bicyclist",
    "motorcyclist",
    "building",
    "other-structure",
    "vegetation",
    "trunk",
    "terrain",
    "traffic-sign",
    "pole",
    "traffic-cone",
    "fence",
    "garbage-can",
    "electric-box",
    "table",
    "chair",
    "bench",
    "other-object",
]


def evaluate(model, dataloader, config, save_path=None, device=0):
    """
    Function to evaluate the performances of a downstream training.
    It prints the per-class IoU, mIoU and fwIoU.
    """
    model.eval()
    with torch.no_grad():
        i = 0
        full_predictions = []
        ground_truth = []
        # lidar_token = []
        for batch in tqdm(dataloader):
            sparse_input = SparseTensor(batch["sinput_F"], batch["sinput_C"], device=device)
            # sparse_input = SparseTensor(batch["sinput_F"], batch["sinput_C"]).cuda()
            output_points = model(sparse_input).F
            if config["ignore_index"]:
                output_points[:, config["ignore_index"]] = -1e6

            torch.cuda.empty_cache()
            preds = output_points.argmax(1).cpu()
            offset = 0
            for j, lb in enumerate(batch["len_batch"]):
                inverse_indexes = batch["inverse_indexes"][j]
                predictions = preds[inverse_indexes + offset]

                # remove the ignored index entirely
                full_predictions.append(predictions)
                ground_truth.append(deepcopy(batch["evaluation_labels"][j]))
                offset += lb
            i += j
            # lidar_token.extend(batch['lidar_token'])

        m_IoU, fw_IoU, per_class_IoU = compute_IoU(
            torch.cat(full_predictions),
            torch.cat(ground_truth),
            config["model_n_out"],
            ignore_index=0,
        )
        CLASSES_NAME_MAP = None
        print("Per class IoU:")
        if config["dataset"].lower() == "nuscenes":
            CLASSES_NAME_MAP = CLASSES_NUSCENES
            # print(
            #     *[
            #         f"{a:20} - {b:.3f}"
            #         for a, b in zip(CLASSES_NUSCENES, (per_class_IoU).numpy())
            #     ],
            #     sep="\n",
            # )
        elif config["dataset"].lower() == "kitti":
            CLASSES_NAME_MAP = CLASSES_KITTI
            # print(
            #     *[
            #         f"{a:20} - {b:.3f}"
            #         for a, b in zip(CLASSES_KITTI, (per_class_IoU).numpy())
            #     ],
            #     sep="\n",
            # )
        elif config['dataset'].lower() == 'waymo':
            CLASSES_NAME_MAP = CLASSES_WAYMO
            # print(
            #     *[
            #         f"{a:20} - {b:.3f}"
            #         for a, b in zip(CLASSES_WAYMO, (per_class_IoU).numpy())
            #     ],
            #     sep="\n",
            # )
        elif config['dataset'].lower() == 'synth4d':
            CLASSES_NAME_MAP = CLASSES_SYNTH4D
            # print(
            #     *[
            #         f"{a:20} - {b:.3f}"
            #         for a, b in zip(CLASSES_SYNTH4D, (per_class_IoU).numpy())
            #     ],
            #     sep="\n",
            # )
        elif config['dataset'].lower() == 'semposs':
            CLASSES_NAME_MAP = CLASSES_SEMPOSS
            # print(
            #     *[
            #         f"{a:20} - {b:.3f}"
            #         for a, b in zip(CLASSES_SEMPOSS, (per_class_IoU).numpy())
            #     ],
            #     sep="\n",
            # )
        elif config['dataset'].lower() == 'semstf':
            CLASSES_NAME_MAP = CLASSES_SEMSTF
            # print(
            #     *[
            #         f"{a:20} - {b:.3f}"
            #         for a, b in zip(CLASSES_SEMSTF, (per_class_IoU).numpy())
            #     ],
            #     sep="\n",
            # )
        elif config['dataset'].lower() == 'rellis3d':
            CLASSES_NAME_MAP = CLASSES_RELLIS3D
            # print(
            #     *[
            #         f"{a:20} - {b:.3f}"
            #         for a, b in zip(CLASSES_RELLIS3D, (per_class_IoU).numpy())
            #     ],
            #     sep="\n",
            # )
        elif config['dataset'].lower() == 'synlidar':
            CLASSES_NAME_MAP = CLASSES_SYNLIDAR
            # print(
            #     *[
            #         f"{a:20} - {b:.3f}"
            #         for a, b in zip(CLASSES_SYNLIDAR, (per_class_IoU).numpy())
            #     ],
            #     sep="\n",
            # )
        else:
            print('Unknown DATASETS!')
            exit(-1)

        print(
            *[
                f"{a:20} - {b:.3f}"
                for a, b in zip(CLASSES_NAME_MAP, (per_class_IoU).numpy())
            ],
            sep="\n",
        )
        print()
        print(f"mIoU: {m_IoU}")
        print(f"fwIoU: {fw_IoU}")

        if config.get('run_dir', None) is not None:
            result_save_path = os.path.join(config["working_dir"], str(config["datetime"]) + '-' + str(config['run_dir']), 'result.txt')
            with open(result_save_path, 'w') as f:
                f.write(f"pretraining_path: {config['pretraining_path']}\n")
                f.writelines([
                        f"{a:20} - {b:.3f}\n"
                        for a, b in zip(CLASSES_NUSCENES, (per_class_IoU).numpy())
                    ]
                )
                f.write('\n')
                f.write(f"mIoU: {m_IoU}\n")
                f.write(f"fwIoU: {fw_IoU}")
            print(f'results save at {result_save_path}.')



        # if save_path is not None:
        #     saved_pkl = dict()
        #     for k, v in zip(lidar_token, full_predictions):
        #         saved_pkl[k] = v
        #     with open(save_path, 'wb') as f:
        #         pickle.dump(saved_pkl, f)
        #     print('results saved at', save_path)

    return m_IoU


def evaluate_spvcnn(model, dataloader, config):
    """
    Function to evaluate the performances of a downstream training.
    It prints the per-class IoU, mIoU and fwIoU.
    """
    model.eval()
    with torch.no_grad():
        i = 0
        full_predictions = []
        ground_truth = []
        for batch in tqdm(dataloader):
            sparse_input = batch['Input_P'].cuda()
            # sparse_input = SparseTensor(batch["sinput_F"], batch["sinput_C"]).cuda()
            output_points = model(sparse_input)
            if config["ignore_index"]:
                output_points[:, config["ignore_index"]] = -1e6

            torch.cuda.empty_cache()
            preds = output_points.argmax(1).cpu()
            offset = 0
            for j, lb in enumerate(batch["len_batch"]):
                inverse_indexes = batch["inverse_indexes"][j]
                predictions = preds[inverse_indexes + offset]

                # remove the ignored index entirely
                full_predictions.append(predictions)
                ground_truth.append(deepcopy(batch["evaluation_labels"][j]))
                offset += lb
            i += j
        m_IoU, fw_IoU, per_class_IoU = compute_IoU(
            torch.cat(full_predictions),
            torch.cat(ground_truth),
            config["model_n_out"],
            ignore_index=0,
        )
        print("Per class IoU:")
        if config["dataset"].lower() == "nuscenes":
            print(
                *[
                    f"{a:20} - {b:.3f}"
                    for a, b in zip(CLASSES_NUSCENES, (per_class_IoU).numpy())
                ],
                sep="\n",
            )
        elif config["dataset"].lower() == "kitti":
            print(
                *[
                    f"{a:20} - {b:.3f}"
                    for a, b in zip(CLASSES_KITTI, (per_class_IoU).numpy())
                ],
                sep="\n",
            )
        elif config['dataset'].lower() == 'waymo':
            print(
                *[
                    f"{a:20} - {b:.3f}"
                    for a, b in zip(CLASSES_WAYMO, (per_class_IoU).numpy())
                ],
                sep="\n",
            )
        elif config['dataset'].lower() == 'synth4d':
            print(
                *[
                    f"{a:20} - {b:.3f}"
                    for a, b in zip(CLASSES_SYNTH4D, (per_class_IoU).numpy())
                ],
                sep="\n",
            )
        print()
        print(f"mIoU: {m_IoU}")
        print(f"fwIoU: {fw_IoU}")

        result_save_path = os.path.join(config["working_dir"], str(config["datetime"]) + '-' + str(config['run_dir']), 'result.txt')
        with open(result_save_path, 'w') as f:
            f.write(f"pretraining_path: {config['pretraining_path']}\n")
            f.writelines([
                    f"{a:20} - {b:.3f}\n"
                    for a, b in zip(CLASSES_NUSCENES, (per_class_IoU).numpy())
                ]
            )
            f.write('\n')
            f.write(f"mIoU: {m_IoU}\n")
            f.write(f"fwIoU: {fw_IoU}")
        print(f'results save at {result_save_path}.')

    return m_IoU
