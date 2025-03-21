# [IEEE TPAMI] Replay Without Saving: Prototype Derivation and Distribution Rebalance for Class-Incremental Semantic Segmentation

This is an official implementation of the paper "Replay Without Saving: Prototype Derivation and Distribution Rebalance for Class-Incremental Semantic Segmentation", accepted by IEEE TPAMI.
📝 [Paper](https://ieeexplore.ieee.org/document/10904177)
🤗 [Hugging Face](https://huggingface.co/jinpeng0528/STAR_TPAMI)

## Installation
### Pre-requisites
This repository has been tested with the following environment:
* CUDA (11.3)
* Python (3.8.13)
* Pytorch (1.12.1)
* Pandas (2.0.3)

### Example conda environment setup
```bash
conda create -n star python=3.8.13
conda activate star
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
conda install pandas==2.0.3
```

## Getting Started

### Datasets

#### PASCAL VOC 2012
We use augmented 10,582 training samples and 1,449 validation samples for PASCAL VOC 2012. You can download the original dataset in [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit). To train our model with augmented samples, please download labels of augmented samples (['SegmentationClassAug'](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip)) and file names (['train_aug.txt'](https://github.com/cvlab-yonsei/DKD/releases/download/v1.0/train_aug.txt)). The structure of data path should be organized as follows:
```bash
└── ./datasets/PascalVOC2012
    ├── Annotations
    ├── ImageSets
    │   └── Segmentation
    │       ├── train_aug.txt
    │       └── val.txt
    ├── JPEGImages
    ├── SegmentationClass
    ├── SegmentationClassAug
    └── SegmentationObject
    
    
```

#### ADE20K
We use 20,210 training samples and 2,000 validation samples for ADE20K. You can download the dataset in [here](http://sceneparsing.csail.mit.edu/). The structure of data path should be organized as follows:
```bash
└── ./datasets/ADE20K
    ├── annotations
    ├── images
    ├── objectInfo150.txt
    └── sceneCategories.txt
```

#### CityScapes
We use 2975 training samples and 500 validation samples for ADE20K. You can download the dataset in [here](https://www.cityscapes-dataset.com/downloads/) ("gtFine_trainvaltest.zip (241MB) [md5]" and "leftImg8bit_trainvaltest.zip (11GB) [md5]"). The structure of data path should be organized as follows:
```bash
└── ./datasets/ADE20K
    ├── leftImg8bit
    ├── gtFine
    ├── fine_train.txt
    └── fine_val.txt
```

### Training
To train STAR-Lite and STAR-Basic models, you can directly run the `.sh` files located in the `./scripts/` directory.

### Testing
#### PASCAL VOC 2012
To evaluate on the PASCAL VOC 2012 dataset, execute the following command:
```Shell
python eval_voc.py --device 0 --test --resume path/to/weight.pth
```

[//]: # (Or, download our pretrained weights and corresponding `config.json` files provided below. Ensure that the config.json file is located in the same directory as the weight file.)

[//]: # ()
[//]: # (| Method<br>&#40;Overlapped&#41; |                                       19-1<br>&#40;2 steps&#41;                                       | 15-5<br>&#40;2 steps&#41; | 15-1<br>&#40;6 steps&#41; | 10-1<br>&#40;11 steps&#41; | 5-3<br>&#40;6 steps&#41; |)

[//]: # (|:-----------------------|:---------------------------------------------------------------------------------------------:|:-----------------:|:-----------------:|:------------------:|:----------------:|)

[//]: # (| STAR-Lite              |                                           [76.61]&#40;&#41;                                           |     [74.86]&#40;&#41;     |     [72.90]&#40;&#41;     |     [64.86]&#40;&#41;      |    [64.54]&#40;&#41;     |)

[//]: # (| STAR-Basic             | [77.02]&#40;&#41; |     [75.80]&#40;&#41;     |     [74.03]&#40;&#41;     |     [66.60]&#40;&#41;      |    [65.65]&#40;&#41;     |)

[//]: # (| STAR-Basic†             | [77.02]&#40;&#41; |     [75.80]&#40;&#41;     |     [74.03]&#40;&#41;     |     [66.60]&#40;&#41;      |    [65.65]&#40;&#41;     |)

[//]: # ()
[//]: # (| Method<br>&#40;Disjoint&#41; | 19-1<br>&#40;2 steps&#41; | 15-5<br>&#40;2 steps&#41; | 15-1<br>&#40;6 steps&#41; | )

[//]: # (|:---------------------|:-----------------:|:-----------------:|:-----------------:|)

[//]: # (| STAR-Lite            |     [76.38]&#40;&#41;     |     [73.48]&#40;&#41;     |     [70.77]&#40;&#41;     |)

[//]: # (| STAR-Basic           |     [76.73]&#40;&#41;     |     [73.79]&#40;&#41;     |     [71.18]&#40;&#41;     |)

[//]: # (| STAR-Basic†    |     [76.73]&#40;&#41;     |     [73.79]&#40;&#41;     |     [71.18]&#40;&#41;     |)


#### ADE20K
To evaluate on the ADE20K dataset, execute the following command:
```Shell
python eval_ade.py --device 0 --test --resume path/to/weight.pth
```

[//]: # (Or, download our pretrained weights and corresponding `config.json` files provided below. Ensure that the config.json file is located in the same directory as the weight file.)

[//]: # ()
[//]: # (| Method<br>&#40;Disjoint&#41; |                                      100-50<br>&#40;2 steps&#41;                                      | 100-10<br>&#40;2 steps&#41; | 50-50<br>&#40;6 steps&#41; | )

[//]: # (|:---------------------|:---------------------------------------------------------------------------------------------:|:-------------------:|:------------------:|)

[//]: # (| STAR-Lite            |                                           [36.39]&#40;&#41;                                           |      [34.91]&#40;&#41;     |     [34.44]&#40;&#41;     |)

[//]: # (| STAR-Basic           | [36.39]&#40;&#41; |      [34.91]&#40;&#41;      |     [34.44]&#40;&#41;    |)


#### CityScapes
To evaluate on the CityScapes dataset, execute the following command:
```Shell
python eval_city.py --device 0 --test --resume path/to/weight.pth
```

[//]: # (Or, download our pretrained weights and corresponding `config.json` files provided below. Ensure that the config.json file is located in the same directory as the weight file.)

[//]: # ()
[//]: # (| Method<br>&#40;Disjoint&#41; |                                       13-6<br>&#40;2 steps&#41;                                       |                                       13-1<br>&#40;2 steps&#41;                                       | )

[//]: # (|:---------------------|:---------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------:|)

[//]: # (| STAR-Lite            | [36.39]&#40;&#41; | [34.91]&#40;&#41; |)

[//]: # (| STAR-Basic           | [36.39]&#40;&#41; | [34.91]&#40;&#41; |)


## Citation
```
@article{chen2025replay,
  title={Replay Without Saving: Prototype Derivation and Distribution Rebalance for Class-Incremental Semantic Segmentation},
  author={Chen, Jinpeng and Cong, Runmin and Yuxuan, Luo and Ip, Horace and Kwong, Sam},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  year={2025},
  volume={},
  number={},
  pages={1-18},
}
```

## Acknowledgements
* This code is based on [DKD](https://github.com/cvlab-yonsei/DKD) ([2022-NeurIPS] Decomposed Knowledge Distillation for Class-Incremental Semantic Segmentation).
* This template is borrowed from [pytorch-template](https://github.com/victoresque/pytorch-template).
