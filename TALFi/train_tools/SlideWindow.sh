echo "start training"
CUDA_VISIBLE_DEVICES=$1,$2 python3 TALFi/utils/train_SL.py configs/SL.yaml

echo "start detecting..."
CUDA_VISIBLE_DEVICES=$1 python3 TALFi/utils/test_SL.py configs/SL.yaml

echo "start eval..."
python3 TALFi/utils/eval.py configs/SL.yaml