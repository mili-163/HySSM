CUDA_VISIBLE_DEVICES=0 python /root/FoldFish/code/cora-diff/eval/evaluation_main.py \
    --ckpt /root/FoldFish/dataset/CoraDiff-Dataset/result/mosi/mr0.7_seed1234/imder-mosi.pth \
    --dataset mosi \
    > /root/FoldFish/code/cora-diff/eval/mosi.log 2>&1