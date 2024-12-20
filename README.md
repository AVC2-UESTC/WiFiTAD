# WiFi Temporal Activity Detection via Dual Pyramid Network
Authors: *Zhendong Liu, Le Zhang, Bing Li, Yingjie Zhou, Chengzheng Hua and Ce Zhu*

## Abstract
We address the challenge of WiFi-based temporal activity detection and  propose an efficient Dual Pyramid Network that integrates Temporal Signal Semantic Encoders and Local Sensitive Response Encoders. The Temporal Signal Semantic Encoder splits feature learning into high and low-frequency components, using a novel Signed Mask-Attention mechanism to emphasize important areas and downplay unimportant ones, with the features fused using ContraNorm. The Local Sensitive Response Encoder captures fluctuations without learning. These feature pyramids are then combined using a new cross-attention fusion mechanism. We also introduce a dataset with over 2,114 activity segments across 553 WiFi CSI samples, each lasting around 85 seconds. Extensive experiments show our method outperforms challenging baselines. [[paper](https://github.com/AVC2-UESTC/WiFiTAD/blob/main/mainPaper.pdf)] [[appendix](https://github.com/AVC2-UESTC/WiFiTAD/blob/main/Appendix.pdf)] 

 <p align="center">
 <img width="700" src="figures/framework.jpg">
 </p>

## Summary 
- First TAD framework for wireless human action understanding, with an untrimmed WiFi CSI dataset.
- Powerful dual-pyramid encoders and multi-level cross-attention feature fusion.
- Easily extansible to other signal modalities.

## Performance

![](figures/performance.png)

## Getting Started
### Dependencies & Installation
We recommend to install Python 3.8 and pytorch 1.12.1: 

`pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116`

Other packages required to support this project can be installed by running:

`pip install -r requirements.txt`

build the project package manager: `python3 setup.py develop`

### Data Preparation
WiFi Temporal Activity Detection Dataset: [link]
<!-- (https://drive.google.com/file/d/1gy0ppFtypVTtgBfrFzdMJUbXTb1MbPSK/view?usp=drive_link) -->

### Training and Tnference
Run the traing and inference processes in terminal by: `bash WiFiTAD/train_tools/tools.sh 0,1`

## Citation & Acknowledgment
If you find this project uesful to your research, please use the following BibTex entry.
```
@InProceedings
```

This code is built on AFSD and Actionformer. We greatly express our gratitude for their contributions.
