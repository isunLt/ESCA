import os.path

import torch
import argparse
from downstream.evaluate import evaluate
from utils.read_config import generate_config
from downstream.model_builder import make_model
from downstream.dataloader_kitti import make_data_loader as make_data_loader_kitti
from downstream.dataloader_nuscenes_c import make_data_loader as make_data_loader_nuscenes
from downstream.dataloader_waymo import make_data_loader as make_data_loader_waymo
import numpy as np
from nuscenes import nuscenes
import math

def calculate_mCE(model, baseline):
    score = [model[key][0] for key in model.keys() if key != 'clean']
    score = 100 - np.array(score)
    score_baseline = [baseline[key][0] for key in baseline.keys() if key != 'clean']
    score_baseline = 100 - np.array(score_baseline)
    CE = score / score_baseline

    mCE = np.mean(CE)
    print("mCE: {:.2f}%.".format(mCE * 100))
    CE = np.round(CE * 100, 2)
    print("CE: {}.".format(CE))

    return mCE, CE

def calculate_mRR(model):
    score = [model[key][0] for key in model.keys() if key != 'clean']
    score = np.array(score)
    RR = score / model['clean'][0]

    mRR = np.mean(RR)
    print("mRR: {:.2f}%.".format(mRR * 100))
    RR = np.round(RR * 100, 2)
    print("RR: {}.".format(RR))

    return mRR, RR

MinkUNet_18_cr10_baseline_nusc_seg = {
  # type,             mIoU,
  'clean':           [75.76],
  'fog':             [53.64],
  'wet_ground':      [73.91],
  'snow':            [40.35],
  'motion_blur':     [73.39],
  'beam_missing':    [68.54],
  'crosstalk':       [26.58],
  'incomplete_echo': [63.83],
  'cross_sensor':    [50.95],
}

CORRUPT_TYPE = ['fog', 'wet_ground', 'snow', 'motion_blur', 'beam_missing', 'crosstalk', 'incomplete_echo', 'cross_sensor']
CORRUPT_LEVEL = ['light', 'moderate', 'heavy']

def main():
    """
    Code for launching the downstream evaluation
    """
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument(
        "--cfg_file", type=str, default=None, help="specify the config for training"
    )
    parser.add_argument(
        "--resume_path", type=str, default=None, help="provide a path to resume an incomplete training"
    )
    parser.add_argument(
        "--dataset", type=str, default=None, help="Choose between nuScenes and KITTI"
    )
    parser.add_argument(
        "--save_path", type=str, default=None, help="Results save path"
    )
    parser.add_argument(
        "--device", type=int, default=0, help="Results save path"
    )
    args = parser.parse_args()
    if args.cfg_file is None and args.dataset is not None:
        if args.dataset.lower() == "kitti":
            args.cfg_file = "config/semseg_kitti.yaml"
        elif args.dataset.lower() == "nuscenes":
            args.cfg_file = "config/eval_nuscenes_c.yaml"
        elif args.dataset.lower() == 'waymo':
            args.cfg_file = "config/semseg_waymo.yaml"
        else:
            raise Exception(f"Dataset not recognized: {args.dataset}")
    elif args.cfg_file is None:
        args.cfg_file = "config/semseg_nuscenes.yaml"

    config = generate_config(args.cfg_file)
    if args.resume_path:
        config['resume_path'] = args.resume_path

    print("\n" + "\n".join(list(map(lambda x: f"{x[0]:20}: {x[1]}", config.items()))))

    print("Creating the model")
    model = make_model(config, config["pretraining_path"]).to(args.device)
    checkpoint = torch.load(config["resume_path"], map_location='cpu')
    if "config" in checkpoint:
        for cfg in ("voxel_size", "cylindrical_coordinates"):
            assert checkpoint["config"][cfg] == config[cfg], (
                f"{cfg} is not consistant.\n"
                f"Checkpoint: {checkpoint['config'][cfg]}\n"
                f"Config: {config[cfg]}."
            )
    try:
        model.load_state_dict(checkpoint["model_points"])
    except KeyError:
        weights = {
            k.replace("model.", ""): v
            for k, v in checkpoint["state_dict"].items()
            if k.startswith("model.")
        }
        model.load_state_dict(weights)

    print("Creating the loaders")

    ret_dict = dict()
    for corrupt_type in CORRUPT_TYPE:
        ret_dict[corrupt_type] = dict()
        for corrupt_level in CORRUPT_LEVEL:
            ret_dict[corrupt_type][corrupt_level] = 0.

    if config["dataset"].lower() == "nuscenes":
        phase = "verifying" if config['training'] in ("parametrize", "parametrizing") else "val"
        nusc = nuscenes.NuScenes(dataroot='datasets/nuscenes/', version='v1.0-trainval', verbose=False)
        for corrupt_type in CORRUPT_TYPE:
            for corrupt_level in CORRUPT_LEVEL:
                if not os.path.exists(os.path.join(config['corrupt_dataroot'], corrupt_type, corrupt_level)):
                    continue
                val_dataloader = make_data_loader_nuscenes(
                    config, phase=phase, num_threads=config["num_threads"], cached_nuscenes=nusc, corrupt_type=corrupt_type, level=corrupt_level
                )
                mIoU = evaluate(model, val_dataloader, config, args.save_path, device=args.device)
                ret_dict[corrupt_type][corrupt_level] = mIoU
                print('%s_%s_mIoU: %.4f' % (corrupt_type, corrupt_level, mIoU))
        print(ret_dict)
        ret_dict['clean'] = [75.85]
        for corrupt_type in CORRUPT_TYPE:
            tmp = []
            for corrupt_level in CORRUPT_LEVEL:
                tmp.append(ret_dict[corrupt_type][corrupt_level])
            ret_dict[corrupt_type] = [sum(tmp) / len(tmp) * 100]
        print(f"mCE={calculate_mCE(ret_dict, baseline=MinkUNet_18_cr10_baseline_nusc_seg)}")
        print(f"mRR={calculate_mRR(ret_dict)}")
        # tmp_dict = dict()
        # for corrupt_type in CORRUPT_TYPE:
        #
        #     tmp_dict[corrupt_type] = list(ret_dict[corrupt_type].values())

    # elif config["dataset"].lower() == "kitti":
    #     val_dataloader = make_data_loader_kitti(
    #         config, "val", num_threads=config["num_threads"]
    #     )
    # elif config['dataset'].lower() == 'waymo':
    #     val_dataloader = make_data_loader_waymo(
    #         config, 'val', num_threads=config['num_threads']
    #     )
    # else:
    #     raise Exception(f"Dataset not recognized: {args.dataset}")




if __name__ == "__main__":
    main()
