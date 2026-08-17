# Exploring the Untouched Sweeps for Conflict-Aware 3D Perception Pretraining

Official PyTorch implementation of the paper *Exploring the Untouched Sweeps for Conflict-Aware 3D Perception Pretraining*.

## Dependencies
Please install the required required packages. Some libraries used in this project, including MinkowskiEngine and Pytorch-lightning are known to have a different behavior when using a different version; please use the exact versions specified in `requirements.txt`.

## Datasets
1. Download nuScenes dataset from the official [link](https://www.nuscenes.org/lidar-segmentation) and put it in `{project_root}/datasets/nuscenes`.
2. Download the superpixels `dino_mask_png.tgz` from the [BAIDU](https://pan.baidu.com/s/1-W-8ZinMkNZmVk35mS8RFg?pwd=bj4d) and unzip the file. Set the `superpixels_path` in the file `configs/slidr_minkunt.yaml` to the unzip path.
3. Download the LiDAR-Image pair `sweeps_flitered_by_dino_mIoU.pkl` selected by our `VFM-Driven Sample Exploring(VSE)` module from [BAIDU](https://pan.baidu.com/s/1-W-8ZinMkNZmVk35mS8RFg?pwd=bj4d) and modify the `sweeps_pair_list_path` in the file `configs/slidr_minkunt.yaml` to the path.
4. To support SemanticKITTI, download it from the official [link](https://semantic-kitti.org/) and put it in `{project_root}/datasets/semantickitti`.
### Optional
If you would like to generate the LiDAR-Image pair `sweeps_flitered_by_dino_mIoU.pkl` by your self, following the steps:
1. Generate the dinov2 masks (in ADE200K label domain) for the images in the official training set of `nuscenes`, including the non-keyframe images.
2. Generate the LiDAR-Image pairs and filtered them by timestamps:
```bash
python3 vse.py
# this will results in sweeps_pairs_filtered_by_mean.pkl
``` 
3. Filtered remaining pairs by dinov2 masks:
```bash
python3 select_sweeps_with_dino.py
# this will generate sweeps_flitered_by_dino_mIoU.pkl
```

## Reproducing the results

### 3D Semantic Segmentation

1. Pretrain the 3D backbone on nuScenes:
```bash
# for minkunet
python pretrain.py --cfg_file config/slidr_minkunet.yaml --run_dir dinov2_supcon_supconintra_vse
# for spvcnn
python pretrain_spvcnn.py --cfg_file config/slidr_spvcnn.yaml --run_dir dinov2_supcon_supconintra_vse
```
2. Fine-tune the 3D backbone using the provided script:
```bash
# for Minkunet
bash downstream_semseg_finetune.sh 0,1,2,3 output/pretrain/nuscenes_minkunet/ddmmyyyy-hhmm-dinov2_supcon_supconintra_vse/model.pt dinov2_supcon_supconintra_vse
# for spvcnn
bash downstream_spvcnn_semseg_finetune.sh 0,1,2,3 output/pretrain/nuscenes_spvcnn/ddmmyyyy-hhmm-dinov2_supcon_supconintra_vse/model.pt dinov2_supcon_supconintra_vse
```

[Opt] To re-evaluate the score of 3d semantic segmentation task, run:

```bash
# for minkunet
python evaluate.py --resume_path="output/semseg/[...]/best_miou_model.pt" --dataset="nuscenes"
# for spvcnn
python evaluate_spvcnn.py --resume_path="output/semseg/[...]/best_miou_model.pt" --dataset="nuscenes"
```

### 3D Panoptic Segmentation
1. Pre-train the 3D backbone Cylinder3D:
```bash
python pretrain_cylinder3D.py --cfg_file config/slidr_cylinder3d.yaml --run_dir dinov2_supcon_supconintra_vse
```
2. Fine-tune the 3D backbone using the provided script:
```bash
sh downstream_panseg_finetune.sh 0,1 output/pretrain/nuscenes_cylinder3d/ddmmyyyy-hhmm-dinov2_supcon_supconintra_vse/model.pt dinov2_supcon_supconintra_vse
```

### 3D Object detection
1. Pre-train the 3D backbone VoxelNet:
```bash
python pretrain.py --cfg_file config/slidr_voxelnet.yaml --run_dir dinov2_supcon_supconintra_vse
```
2. Fine-tune the VoxelNet using [OpenPCDet](https://github.com/open-mmlab/OpenPCDet). Please refer to the [CSC](https://github.com/chenhaomingbob/CSC).

## Acknowledgment
The codebase is adapted from [SLidR](https://github.com/valeoai/SLidR) and [CSC](https://github.com/chenhaomingbob/CSC).
