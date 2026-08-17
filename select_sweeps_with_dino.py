import numpy as np
import torch
import argparse

import visualize_utils
from downstream.evaluate import evaluate
from utils.read_config import generate_config
from downstream.model_builder import make_model
from pretrain.dataloader_nuscenes_sweeps_find import make_data_loader

from tqdm import tqdm
from utils.metrics import compute_IoU
import pickle

def binary_mask_dice_loss(self, mask_preds, gt_masks):
    """
    Args:
        mask_preds (Tensor): Mask prediction in shape (N1, H, W).
        gt_masks (Tensor): Ground truth in shape (N2, H, W)
            store 0 or 1, 0 for negative class and 1 for
            positive class.

    Returns:
        Tensor: Dice cost matrix in shape (N1, N2).
    """
    mask_preds = mask_preds.reshape((mask_preds.shape[0], -1))
    gt_masks = gt_masks.reshape((gt_masks.shape[0], -1)).float()
    numerator = 2 * torch.einsum("nc,mc->nm", mask_preds, gt_masks)
    denominator = mask_preds.sum(-1)[:, None] + gt_masks.sum(-1)[None, :]
    loss = 1 - (numerator + self.eps) / (denominator + self.eps)
    return loss

def selection_with_dino(dataloader):
    """
    Function to evaluate the performances of a downstream training.
    It prints the per-class IoU, mIoU and fwIoU.
    """
    def _vis(img, idx_list):
        img = img.permute(0, 2, 3, 1).contiguous() * 255.0
        for c_i in range(6):
            im_idx = [c_i + b_i * cam_num for b_i in idx_list]
            for i in im_idx:
                visualize_utils.visualize_img(img[i])

    cam_num = 6
    select_tokens = {
        'top_1': [],
        'top_2': [],
        'top_3': [],
        'top_4': [],
        'top_5': [],
        'top_6': [],
        'top_7': [],
        'top_8': [],
        'top_9': [],
    }
    for batch in tqdm(dataloader):
        images = batch['input_I']
        superpixels = batch['superpixels']
        lidar_token_list = batch['lidar_tokens']
        kf_flag = batch['keyframe_flag_list']
        sample_dict_list = batch['sample_dict_list']
        batch_size = int(images.size(0) / cam_num)
        cost_matrix = np.zeros(shape=(sum(kf_flag), batch_size, cam_num), dtype=np.float32)
        for cam_i in range(cam_num):
            im_idx = [cam_i + b_i * cam_num for b_i in range(batch_size)]
            kf_img = [i for ii, i in enumerate(im_idx) if kf_flag[ii]]
            for k_ii, k_i in enumerate(kf_img):
                sp = superpixels[k_i]
                for b_i, sw_i in enumerate(im_idx):
                    sp_sw = superpixels[sw_i]
                    mIoU, _, _ = compute_IoU(sp_sw.flatten(), sp.flatten(), num_classes=150)
                    cost_matrix[k_ii, b_i, cam_i] = mIoU
        cost_matrix = np.sum(cost_matrix, axis=0)
        cost_matrix = np.sum(cost_matrix, axis=1)
        sort_idx = np.argsort(cost_matrix)
        # _vis(images, sort_idx)
        for s_ii, s_i in enumerate(sort_idx):
            if not kf_flag[s_i] and select_tokens.get('top_' + str(s_ii+1), None) is not None:
                select_tokens['top_' + str(s_ii+1)].append(sample_dict_list[s_i])
    return select_tokens

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
    args = parser.parse_args()
    if args.cfg_file is None and args.dataset is not None:
        if args.dataset.lower() == "kitti":
            args.cfg_file = "config/semseg_kitti.yaml"
        elif args.dataset.lower() == "nuscenes":
            args.cfg_file = "config/semseg_nuscenes.yaml"
        elif args.dataset.lower() == 'waymo':
            args.cfg_file = "config/semseg_waymo.yaml"
        else:
            raise Exception(f"Dataset not recognized: {args.dataset}")
    elif args.cfg_file is None:
        args.cfg_file = "config/sweeps_explore.yaml"

    config = generate_config(args.cfg_file)
    if args.resume_path:
        config['resume_path'] = args.resume_path

    print("\n" + "\n".join(list(map(lambda x: f"{x[0]:20}: {x[1]}", config.items()))))
    print("Creating the loaders")

    SWEEPS_PAIR_SAVE_PATH = '/data2/share/sweeps_flitered_by_dino_mIoU.pkl'
    dataloader = make_data_loader(config, phase='parametrizing', num_threads=16)
    select_dict = selection_with_dino(dataloader)
    with open(SWEEPS_PAIR_SAVE_PATH, 'wb') as f:
        pickle.dump(select_dict, f)
    print('pkl saved at', SWEEPS_PAIR_SAVE_PATH)

if __name__ == "__main__":
    main()
