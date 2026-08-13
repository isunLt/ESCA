#!/bin/bash
# 使用范例:
# sh sem_downstream.sh 0,1 model.pt

# 获取参数
GPU_IDS=$1
MODEL_PATH=$2
EXP_NAME=$3
export CUDA_VISIBLE_DEVICES=$GPU_IDS

################### nuScenes ###################
# 1% fine-tuning
echo "Starting 1% label fine-tuning in nuScenes"
echo "python downstream.py --cfg_file ./config/semseg_nuscenes.yaml --pretraining_path $MODEL_PATH --run_dir $EXP_NAME --dataset_skip 100"
python downstream.py --cfg_file ./config/semseg_nuscenes.yaml --pretraining_path $MODEL_PATH --run_dir $EXP_NAME --dataset_skip 100
# 5% fine-tuning
echo "Starting 5% label fine-tuning in nuScenes"
echo "python downstream.py --cfg_file ./config/semseg_nuscenes.yaml --pretraining_path $MODEL_PATH --run_dir $EXP_NAME --dataset_skip 20"
python downstream.py --cfg_file ./config/semseg_nuscenes.yaml --pretraining_path $MODEL_PATH --run_dir $EXP_NAME --dataset_skip 20
# 10% fine-tuning
echo "Starting 10% label fine-tuning in nuScenes"
echo "python downstream.py --cfg_file ./config/semseg_nuscenes.yaml --pretraining_path $MODEL_PATH --run_dir $EXP_NAME --dataset_skip 10"
python downstream.py --cfg_file ./config/semseg_nuscenes.yaml --pretraining_path $MODEL_PATH --run_dir $EXP_NAME --dataset_skip 10
# 25% fine-tuning
echo "Starting 25% label fine-tuning in nuScenes"
echo "python downstream.py --cfg_file ./config/semseg_nuscenes.yaml --pretraining_path $MODEL_PATH --run_dir $EXP_NAME --dataset_skip 4"
python downstream.py --cfg_file ./config/semseg_nuscenes.yaml --pretraining_path $MODEL_PATH --run_dir $EXP_NAME --dataset_skip 4
# 100% fine-tuning
echo "Starting 100% label fine-tuning in nuScenes"
echo "python downstream.py --cfg_file ./config/semseg_nuscenes.yaml --pretraining_path $MODEL_PATH --run_dir $EXP_NAME --dataset_skip 1"
python downstream.py --cfg_file ./config/semseg_nuscenes.yaml --pretraining_path $MODEL_PATH --run_dir $EXP_NAME --dataset_skip 1
# 100% linear probing
echo "Starting 100% label linear probing in nuScenes"
echo "python ../downstream.py --cfg_file ./config/semseg_nuscenes_lp.yaml --pretraining_path $MODEL_PATH --run_dir $EXP_NAME --dataset_skip 1"
python downstream.py --cfg_file ./config/semseg_nuscenes_lp.yaml --pretraining_path $MODEL_PATH --run_dir $EXP_NAME --dataset_skip 1
################### SemanticKITTI ###################
# 1% fine-tuning
echo "Starting 1% label fine-tuning in SemanticKITTI"
echo "python downstream.py --cfg_file ./config/semseg_kitti.yaml --pretraining_path $MODEL_PATH --run_dir $EXP_NAME"
python downstream.py --cfg_file ./config/semseg_kitti.yaml --pretraining_path $MODEL_PATH --run_dir $EXP_NAME
####
# 100% fine-tuning
echo "Starting 100% label fine-tuning in SemanticKITTI"
echo "python downstream.py --cfg_file ./config/semseg_kitti.yaml --pretraining_path $MODEL_PATH --run_dir $EXP_NAME --dataset_skip 1"
python downstream.py --cfg_file ./config/semseg_kitti.yaml --pretraining_path $MODEL_PATH --run_dir $EXP_NAME --dataset_skip 1

################### Waymo ###################
# 1% fine-tuning
#echo "Starting 1% label fine-tuning in Waymo"
#echo "python downstream.py --cfg_file ./config/semseg_waymo.yaml --pretraining_path $MODEL_PATH --run_dir $EXP_NAME"
#python downstream.py --cfg_file ./config/semseg_waymo.yaml --pretraining_path $MODEL_PATH --run_dir $EXP_NAME
####
# 100% fine-tuning
#echo "Starting 100% label fine-tuning in Waymo"
#echo "python downstream.py --cfg_file ./config/semseg_waymo.yaml --pretraining_path $MODEL_PATH --run_dir $EXP_NAME --dataset_skip 1"
#python downstream.py --cfg_file ./config/semseg_waymo.yaml --pretraining_path $MODEL_PATH --run_dir $EXP_NAME --dataset_skip 1
exit 0
