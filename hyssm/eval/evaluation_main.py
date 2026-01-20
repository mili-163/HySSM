import os
import torch
import numpy as np
import argparse
import sys
sys.path.append("/root/FoldFish/code/cora-diff")

from trains.singleTask.model.imder import IMDER 
from data_loader import MMDataLoader
from utils import MetricsTop

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, required=True, help='Path to best_model.pth (MR=0.7)')
parser.add_argument('--dataset', type=str, default='mosi')
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = 'cuda'

print(f"🚀 Evaluating on {args.dataset} with Fixed Missing Patterns...")
print(f"📂 Model: {args.ckpt}")

# Configuration
config = {
    'dataset_name': args.dataset,
    'use_bert': False,
    'train_mode': 'regression',
    'batch_size': 32,
    'feature_dims': [768, 5, 20], 
    'dst_feature_dim_nheads': [32, 8],
    'nlevels': 4,
    'attn_dropout': 0.1,
    'attn_dropout_a': 0.0,
    'attn_dropout_v': 0.0,
    'relu_dropout': 0.1,
    'embed_dropout': 0.25,
    'res_dropout': 0.1,
    'output_dropout': 0.5,
    'text_dropout': 0.1,
    'attn_mask': True,
    
    'conv1d_kernel_size_l': 3, 
    'conv1d_kernel_size_a': 3,
    'conv1d_kernel_size_v': 3,
    
    'num_classes': 1,
}

# Load Data
dataloader = MMDataLoader(config, num_workers=0)['test']
config['feature_dims'] = dataloader.dataset.args['feature_dims']
print(f"Data Loaded. Dims: {config['feature_dims']}")

# Load Model
model = IMDER(argparse.Namespace(**config))
model.to(device)

# Load Checkpoint
try:
    state_dict = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state_dict)
    print("Model Weights Loaded Successfully.")
except Exception as e:
    print(f"Error loading checkpoint: {e}")
    exit()

model.eval()

# Define Scenarios
scenarios = {
    # Two Modalities Available (One Missing)
    'Miss L ({V, A})': [1, 2],
    'Miss V ({L, A})': [0, 2],
    'Miss A ({L, V})': [0, 1],
    
    # One Modality Available (Two Missing)
    'Miss L, V ({A})': [2],
    'Miss L, A ({V})': [1],
    'Miss V, A ({L})': [0],
    
    # Full Modality
    'Full ({L, V, A})': [0, 1, 2] 
}

metrics = MetricsTop('regression').getMetics(args.dataset)

print("\n" + "="*55)
print(f"{'Scenario':<20} | {'ACC2 / F1 / ACC7':<30}")
print("="*55)

with torch.no_grad():
    for name, modal_idx in scenarios.items():
        y_pred, y_true = [], []
        
        for batch_data in dataloader:
            vision = batch_data['vision'].to(device)
            audio = batch_data['audio'].to(device)
            text = batch_data['text'].to(device)
            labels = batch_data['labels']['M'].to(device).view(-1, 1)
            
            zs = {k: v.to(device) for k, v in batch_data['zs'].items()}
            zp = {k: v.to(device) for k, v in batch_data['zp'].items()}
            
            # Call Model with Force Modal
            outputs = model(text, audio, vision, 
                          zs=zs, zp=zp,
                          force_modal_idx=modal_idx)
            
            y_pred.append(outputs['M'].cpu())
            y_true.append(labels.cpu())
            
        pred = torch.cat(y_pred)
        true = torch.cat(y_true)
        res = metrics(pred, true)
        
        # [FORMATTED OUTPUT] ACC2 / F1 / ACC7
        acc2 = res['Acc_2'] * 100
        f1 = res['F1_score'] * 100
        acc7 = res['Acc_7'] * 100
        
        print(f"{name:<20} | {acc2:.1f} / {f1:.1f} / {acc7:.1f}")

print("="*55)