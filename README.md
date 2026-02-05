# HySSM

Hybrid State Space Model for Incomplete Multimodal Learning

Semantic-first Completion framework:
- Semantic Planning: Markovian semantic transitions
- Evidence Retrieval: Evidence-grounded trajectory retrieval
- Cross-Space Manifold Projection: Discrete to continuous transformation
- Drift-guided Diffusion: Modality completion conditioned on semantic plans

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

For CUDA support, install PyTorch with CUDA separately.

## Data Preparation

Place your data files in the following structure:

```
dataset/
  MOSI/
    mosi_unified.pt
  MOSEI/
    mosei_unified.pt
```

Or set environment variables `MOSI_DATA_PATH` and `MOSEI_DATA_PATH`.

## Usage

### Training

```bash
python train.py
```

API usage:

```python
from run import HySSM_run

HySSM_run(
    model_name='hyssm',
    dataset_name='mosi',
    seeds=[1111, 1112, 1113, 1114, 1115],
    mr=0.1,  # missing rate
    model_save_dir="./checkpoints",
    res_save_dir="./result",
    log_dir="./log",
    gpu_ids=[0]
)
```

### Evaluation

```bash
python evaluation/evaluation_main.py --ckpt <checkpoint_path> --dataset mosi --gpu 0
```

## Directory Structure

- `components/`: Semantic planning, evidence retrieval, drift-guided diffusion
- `training/`: Training code and model definitions
- `config/`: Configuration files
- `checkpoints/`: Model checkpoints
- `result/`: Training results
- `log/`: Training logs
- `feature_extraction/`: Feature extraction
- `evaluation/`: Evaluation scripts
- `figures/`: Visualization

Configuration in `config/config.json`.
