# WiFi-Temporal-Activity-Detection
A model aims to handle wifi temporal activity Detection

## Abstract
This model tackles wifi activity detection task with an end-to-end, anchor-free network.

## Getting Started

### Environment



We recommend to install Python 3.8 and pytorch 1.9.1: 

`pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html`

Other packages required to support this project can be installed by running:

`pip install -r requirements.txt`

### Setup

build the project package manager: `python3 setup.py develop`

### Training and Tnference
you can run the TALFi in terminal by: `bash TALFi/train_tools/TALFi.sh 0,1,2,3`

as well run the Slide Window Style Classify model by: `bash TALFi/utils/SlideWindow.sh 0`

### Data Preparation
our dataset is available in :``

## Citation
