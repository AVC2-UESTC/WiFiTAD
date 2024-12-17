echo "start training"
CUDA_VISIBLE_DEVICES=$1,$2 python3 TAD/utils/train.py configs/WiFiTAD.yaml

echo "start detecting..."
CUDA_VISIBLE_DEVICES=$1 python3 TAD/utils/test.py configs/WiFiTAD.yaml

echo "start eval..."
python3 TAD/utils/eval.py configs/WiFiTAD.yaml