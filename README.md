# [AAAI25] WiFi Temporal Activity Detection via Dual Pyramid Network

Authors: *Zhendong Liu, Le Zhang, Bing Li, Yingjie Zhou, Zhenghua Chen and Ce Zhu*

## Abstract

We address the challenge of WiFi-based temporal activity detection and  propose an efficient Dual Pyramid Network that integrates Temporal Signal Semantic Encoders and Local Sensitive Response Encoders. The Temporal Signal Semantic Encoder splits feature learning into high and low-frequency components, using a novel Signed Mask-Attention mechanism to emphasize important areas and downplay unimportant ones, with the features fused using ContraNorm. The Local Sensitive Response Encoder captures fluctuations without learning. These feature pyramids are then combined using a new cross-attention fusion mechanism. We also introduce a dataset with over 2,114 activity segments across 553 WiFi CSI samples, each lasting around 85 seconds. Extensive experiments show our method outperforms challenging baselines. [[paper](https://github.com/AVC2-UESTC/WiFiTAD/blob/main/mainPaper.pdf)] [[appendix](https://github.com/AVC2-UESTC/WiFiTAD/blob/main/Appendix.pdf)] 

 <p align="center">
 <img width="700" src="figures/1741098982203.png">
 </p>

## Summary 

- First TAD framework for wireless Temporal Activity Detection (also referred to as Temporal Activity Localization), with an untrimmed WiFi CSI dataset.
- Powerful dual-pyramid encoders and multi-level cross-attention feature fusion.
- Easily extansible to other signal modalities.

## Performance

![](figures/performance.png)

## Getting Started

### Dependencies & Installation

Use conda or venv to manage your environment, for example:

```bash
conda create -n wifitad python=3.8
conda activate wifitad
```

To install dependencies and build the project, run:

```bash
pip install .
```

Note that the original was tested using:

```bash
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

### Dataset Downloads

WiFi Temporal Activity Detection Dataset: [link](https://drive.google.com/file/d/1gy0ppFtypVTtgBfrFzdMJUbXTb1MbPSK/view?usp=drive_link)

### Training and Tnference Example

Run the traing and inference processes in terminal by: 

```bash
bash TAD/train_tools/tools.sh 0,1
```

### Notes

You may tune the hyperparameters of NMS to get wider range of TAD results.

## Citation & Acknowledgment

If you find the paper and its code uesful to your research, please use the following BibTex entry.

```bibtex
@article{Liu_Zhang_Li_Zhou_Chen_Zhu_2025, title={WiFi CSI Based Temporal Activity Detection via Dual Pyramid Network}, volume={39}, url={https://ojs.aaai.org/index.php/AAAI/article/view/32035}, DOI={10.1609/aaai.v39i1.32035}, number={1}, journal={Proceedings of the AAAI Conference on Artificial Intelligence}, author={Liu, Zhendong and Zhang, Le and Li, Bing and Zhou, Yingjie and Chen, Zhenghua and Zhu, Ce}, year={2025}, month={Apr.}, pages={550-558} }
```

This code is built on AFSD and Actionformer. We greatly express our gratitude for their contributions.
