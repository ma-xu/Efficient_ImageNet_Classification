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

## Training & Testing
We provide two training strategies: step_lr schedular and cosine_lr schedular in [main_step.py](https://github.com/13952522076/Efficient_ImageNet_Classification/blob/master/main_step.py) and [main_cosine.py](https://github.com/13952522076/Efficient_ImageNet_Classification/blob/master/main_cosine.py) respectively.

The training models (last one and best one) and the log file  are saved in "checkpoints/imagenet/`model_name`" by default.
***

I personally suggest to manually setup the path to imagenet dataset in [main_step.py (line 49)](https://github.com/13952522076/Efficient_ImageNet_Classification/blob/f6218ccc0992458909460c095795d9aca3e48c18/main_step.py#L49) 
and [main_cosine.py (line 50)](https://github.com/13952522076/Efficient_ImageNet_Classification/blob/f6218ccc0992458909460c095795d9aca3e48c18/main_cosine.py#L50).
Replace the default value to your real PATH.

Or you can add a parameter `--data` in the following training command.


**For the step learning rate schedular, run follwing commands**
```Bash
# change the parameters accordingly if necessary
# e.g, If you have 4 GPUs, set the nproc_per_node to 4. If you want to train with 32FP, remove ----fp16.
python3 -m torch.distributed.launch --nproc_per_node=8 main_step.py -a old_resnet50 --fp16 --b 32
```
**For the cosine learning rate schedular, run follwing commands**
```Bash
# change the parameters accordingly if necessary
python3 -m torch.distributed.launch --nproc_per_node=8 main_cosine.py -a old_resnet18 --b 64 --opt-level O0
```
## Add New Models
Please follow the same coding style in [models/resnet.py](https://github.com/13952522076/Efficient_ImageNet_Classification/blob/master/models/resnet.py). 

1. Add a new model file in folder [models](https://github.com/13952522076/Efficient_ImageNet_Classification/tree/master/models)
2. Import the model file in model package, say [models/__init__.py](https://github.com/13952522076/Efficient_ImageNet_Classification/blob/master/models/__init__.py)

## Acknowledgements
This implementation is built upon [ImageNet demo](https://github.com/pytorch/examples/tree/master/imagenet) and [PytorchInsight](https://github.com/implus/PytorchInsight). 

Many thanks to [Xiang Li](http://implus.github.io/) for his great work. 
