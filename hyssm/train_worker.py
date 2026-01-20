import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--mr', type=float, required=True, help='Missing rate (0.0 - 0.7)')
parser.add_argument('--gpu', type=str, default='1', help='GPU ID')
parser.add_argument('--dataset', type=str, default='mosi', help='Dataset name')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

from run import IMDER_run

base_seed = 2026
seeds_list = [base_seed]
dataset_name = args.dataset
mr = args.mr

# 统一路径管理
save_dir_base = f"/root/FoldFish/dataset/CoraDiff-Dataset/result/{dataset_name}/mr{mr}_seed{base_seed}"

print(f"🚀 [Worker Start] MR={mr}, Dataset={dataset_name}, Seeds={seeds_list}, GPU={args.gpu}")


try:
    IMDER_run(
        model_name='imder',
        dataset_name=dataset_name,
        seeds=seeds_list,
        mr=mr,
        model_save_dir=save_dir_base,
        res_save_dir=save_dir_base,
        log_dir=save_dir_base,
        resume_from_checkpoint=False,
        num_workers=0 
    )
    print(f"✅ [Worker Done] MR={mr} Finished successfully.")
except Exception as e:
    print(f"❌ [Worker Failed] MR={mr} crashed.")
    raise e