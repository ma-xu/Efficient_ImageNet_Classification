# Efficient ImageNet Classification

:rocket: Training ImageNet in 8 hours.

This repo provides an efficient implementation of ImageNet classification, based on PyTorch, DALI, and Apex.

If any questions, please create an issue or contact [me](https://13952522076.github.io/) at <xuma@my.unt.edu>

## Getting Start
### Installation

 __1. Download repo__
 
```Bash
git clone https://github.com/13952522076/Efficient_ImageNet_Classification.git
cd Efficient_ImageNet_Classification
```

__2. Requirements__

- Python3.6
- PyTorch 1.3+
- CUDA 10+
- GCC 5.0+
```Bash
pip install -r requirements.txt
```
__3. Install DALI and Apex__

DALI Installation:
```Bash
cd ~
# For CUDA10
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-tf-plugin-cuda100
# or
# For CUDA11
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-tf-plugin-cuda110
```
For more details, please see [Nvidia DALI installation](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/installation.html).


Apex Installation:
```Bash
cd ~
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
For more details, please see [Apex](https://github.com/NVIDIA/apex) or [Apex Full API documentation](https://nvidia.github.io/apex/).


<!--__Prepare ImageNet dataset__-->

<!--```Bash-->
<!--cd ~-->
<!--cd Efficient_ImageNet_Classification-->
<!--mkdir data-->
<!--cd data-->
<!--# Replace PATH_TO_ImageNet to your ImageNet dataset path-->
<!--ln -s PATH_TO_ImageNet imagenet-->
<!--```-->

## Training
We provide two training strategies: step_lr schedular and cosine_lr schedular in [main_step.py](https://github.com/13952522076/Efficient_ImageNet_Classification/blob/master/main_step.py) and [main_cosine.py](https://github.com/13952522076/Efficient_ImageNet_Classification/blob/master/main_cosine.py) respectively.



## Acknowledgements

