# DiffMamba: Gated Differential Spectral-Spatial Mamba for Few-Shot Hyperspectral Image Classification

## Introduction

* We propose a novel dual-branch network, named DiffMamba, for robust few-shot HSI classification. Specifically, we integrate a Self-Gating Mechanism
into the Mamba units to dynamically recalibrate feature responses and suppress irrelevant noise. Besides, we design a Differen-
tial Spectral-Spatial Mamba module and a Gated Spatial Mamba module to capture spectral fingerprints and long-range spatial dependencies.
* Experimental results conducted on three public datasets demonstrate the effectiveness
and superiority of the proposed model for few-shot HSI classification.

## Getting Started

### Installation

```sh
conda create -n DiffMamba_env python=3.9
conda activate DiffMamba_env
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install packaging==24.0
pip install triton==2.2.0
pip install mamba-ssm==1.2.0
pip install spectral
pip install scikit-learn==1.4.1.post1
pip install calflops
```

### Data Preparation
The dataset can download [BaiduNetdisk](https://pan.baidu.com/s/1dP-Gen8ZRO6yLuTUTUE7GA?pwd=xj8i ).

```
data
└── UP/
    ├── PaviaU.mat 
    └── PaviaU_gt.mat
    ...
└── Houston/
    ├── Houston.mat 
    └── Houston_GT.mat
    ...
└── HanChuan/
    ├── WHU_Hi_HanChuan.mat 
    └── WHU_Hi_HanChuan_gt.mat
```



**Training:**
```
python train_DiffMamba.py --dataset_index 0
python train_DiffMamba.py --dataset_index 1
python train_DiffMamba.py --dataset_index 2
```
