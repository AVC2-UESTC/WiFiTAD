echo "start training"
CUDA_VISIBLE_DEVICES=$1,$2 python TAD/utils/train.py configs/WiFiTAD.yaml

echo "start detecting..."
CUDA_VISIBLE_DEVICES=$1 python TAD/utils/test.py configs/WiFiTAD.yaml

echo "start eval..."
python TAD/utils/eval.py configs/WiFiTAD.yaml