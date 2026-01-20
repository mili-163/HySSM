#!/bin/bash

# 指定使用的 GPU ID (根据您的情况修改，例如 "1" 或 "0")
GPU_ID="1"
DATASET="mosi"

echo "==================================================="
echo "🚀 Starting Parallel Training on GPU ${GPU_ID}..."
echo "==================================================="

# 启动任务放入后台 (&)
# 每个任务会依次跑完 5 个 Seed，但这 8 个 MR 是同时进行的
python train_worker.py --mr 0.6 --gpu $GPU_ID --dataset $DATASET &
python train_worker.py --mr 0.5 --gpu $GPU_ID --dataset $DATASET &
wait
sleep 10 

python train_worker.py --mr 0.4 --gpu $GPU_ID --dataset $DATASET &
python train_worker.py --mr 0.3 --gpu $GPU_ID --dataset $DATASET &
wait
sleep 10 # 稍微冷却一下，释放内存碎片


python train_worker.py --mr 0.2 --gpu $GPU_ID --dataset $DATASET &
python train_worker.py --mr 0.1 --gpu $GPU_ID --dataset $DATASET &
wait
sleep 10

echo "✅ All 8 tasks launched in background!"
echo "⏳ Waiting for completion..."
echo "Running on PID(s): $(jobs -p)"

# 等待所有后台任务结束
wait

echo "🎉 All experiments finished!"