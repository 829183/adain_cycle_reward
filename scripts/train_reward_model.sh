#!/bin/bash

set -e

SCRIPT_DIR="/home/luo/Downloads/CV/final_project/adain_cycle_reward"
cd "$SCRIPT_DIR"
echo "当前工作目录: $(pwd)"

CONDA_ENV_NAME="acrwd"
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_NAME"
echo "已激活 Conda 环境: $CONDA_ENV_NAME"

DATA_JSON="./data/cycle_scores.json"
if [ ! -f "$DATA_JSON" ]; then
    echo "错误: $DATA_JSON 不存在！请先运行 ./generate_cycle_scores.sh"
    exit 1
fi


MODEL_SAVE_PATH="./models/reward_model_resnet18.pth"
echo "开始训练 Reward Model (ResNet-18)..."
python train_reward_model.py \
  --data_json "$DATA_JSON" \
  --epochs 3 \
  --batch_size 32 \
  --lr 1e-4 \
  --save_path "$MODEL_SAVE_PATH"

echo "训练完成！模型保存在: $MODEL_SAVE_PATH"