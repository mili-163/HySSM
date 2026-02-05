import os
from pathlib import Path
# set CUDA_VISIBLE_DEVICES before importing torch (macOS 上使用 CPU，设置为空)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from run import HySSM_run

# 使用相对路径
base_dir = Path(__file__).parent
result_dir = base_dir / "result"

seed=1234
mr=0.7
dataset_name='mosi'
HySSM_run(model_name='hyssm',
           dataset_name=dataset_name,
           seeds=[seed, seed+1, seed+2, seed+3, seed+4],
           mr=mr,
           model_save_dir=str(result_dir / f"{dataset_name}/mr{mr}_seed{seed}"),
           res_save_dir=str(result_dir / f"{dataset_name}/mr{mr}_seed{seed}"),
           log_dir=str(result_dir / f"{dataset_name}/mr{mr}_seed{seed}"),
           resume_from_checkpoint=False)