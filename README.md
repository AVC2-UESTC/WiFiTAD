# WiFi-Temporal-Activity-Localization
A model aims to handle wifi temporal activity localization

## Getting Started

### Environment
The following packages are required to run this project:
-Python 3.8
- pytorch == 1.9.1
- tqdm
- pandas
- jiblib
- pyyaml

### Setup
use this command to install pytorch 1.9.1 : `pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html`
use this command to build the project package manager: `python3 setup.py develop`

### Training and Tnference
you can run the TALFi in terminal by: `bash TALFi/utils/TALFi.sh 0,1,2,3`

as well run the Slide Window Style Classify model by: `bash TALFi/utils/SlideWindow.sh 0`

### Data Preparation
