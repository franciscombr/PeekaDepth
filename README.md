# RGBD-Align

**Contrastive Fusion for RGB–Depth Visual Perception**

## Overview
RGBD-Align is a PyTorch-based framework for learning modality-invariant embeddings by aligning RGB images and depth maps using contrastive learning. By training projection heads on frozen pretrained encoders, it enables robust, label-efficient fusion for downstream tasks like scene classification and semantic segmentation.

## Features
- Cross-modal NT-Xent contrastive loss between RGB and depth representations
- Configurable data pipeline for paired RGB–Depth augmentations
- Modular training and evaluation scripts for classification and segmentation benchmarks
- Experiment tracking with TensorBoard or Weights & Biases

## Installation
1. Clone the repo:
   ~~~bash
   git clone https://github.com/yourusername/RGBD-Align.git
   cd RGBD-Align
   ~~~
2. Set up environment:
   ~~~bash
   conda env create -f environment.yml  # or use requirements.txt
   conda activate rgbd-align
   ~~~
3. Download NYUv2 dataset and place under `data/raw/` (see `data/README.md` for details).

## Usage
- **Training**:
  ~~~bash
  python src/train.py --config config/contrastive.yaml
  ~~~
- **Evaluation**:
  ~~~bash
  python src/evaluate.py --task classification --checkpoint experiments/week2_contrastive/results/model.pt
  ~~~