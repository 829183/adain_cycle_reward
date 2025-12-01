#!/bin/bash

set -e

SCRIPT_DIR="/home/luo/Downloads/CV/final_project/adain_cycle_reward"
cd "$SCRIPT_DIR"
echo "当前工作目录: $(pwd)"

CONDA_ENV_NAME="acrwd"
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_NAME"
echo "已激活 Conda 环境: $CONDA_ENV_NAME"

CONTENT_DIR="./data/content"
STYLE_DIR="./data/style"
OUTPUT_JSON="./data/cycle_scores.json"
STYLIZED_DIR="./data/stylized"

echo "开始生成 stylized 图像和 reward..."
python generate_cycle_scores.py \
  --content_dir "$CONTENT_DIR" \
  --style_dir "$STYLE_DIR" \
  --output_json "$OUTPUT_JSON" \
  --alphas 0.3 0.5 0.7 0.9 1.0 \
  --max_samples 2000

echo "生成完成！结果保存在: $OUTPUT_JSON"