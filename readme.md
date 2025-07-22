
<div align="center">

<h1>Spectral Structure-Aware Initialization and Probability-Consistent Self-Training for Cross-scene Hyperspectral Image Classification</h1>

<h2>IEEE Geoscience and Remote Sensing Letters</h2>


[Junye Liang](https://scholar.google.com/citations?user=pOXE8p8AAAAJ&hl=zh-CN&oi=ao)<sup>1</sup>, [Jiaqi Yang](https://scholar.google.com/citations?hl=zh-CN&user=cQAAdBYAAAAJ)<sup>2 †</sup>, [Rong Liu](https://github.com/liurongwhm)<sup>1 †</sup>, [Quanwei Liu](https://scholar.google.com/citations?user=E-loHKYAAAAJ&hl=zh-CN&oi=ao)<sup>3</sup>, [Peng Zhu](https://scholar.google.com/citations?hl=zh-CN&user=iao5Lp0AAAAJ)<sup>4</sup>

<sup>1</sup> Sun Yat-sen University, <sup>2</sup> Wuhan University,  <sup>3</sup> James Cook University, <sup>4</sup> The University of Hong Kong.

<sup>†</sup> Corresponding author

</div>


# 🌞 Overview

**Spectral Structure-Aware Initialization and Probability-Consistent Self-Training (S2PST)** is a novel framework for cross-scene HSI classification. The framework employs batch nuclear-norm maximization to constrain the probability responses of TD outputs, implicitly aligning feature distributions between SD and TD. To enhance the model’s robustness and spectral feature representation ability, we introduce a spectral structure-aware initialization method that integrates the strengths of traditional machine learning and deep learning. Furthermore, to mitigate the model’s bias toward SD training data, we propose a self-supervised training strategy that dynamically incorporates pseudo-labeled TD samples into the training process by comparing the similarity of high-confidence samples in the probability space between SD and TD.</a>


<p align="center">
<img src=/figure/S2PST.png width="80%">
</P>

<div align='center'>

**Figure 1. Framework of S2PST.**

</div>
<br>

Extensive experiments are conducted on the Houston, HyRANK, and Pavia datasets, and compared with several state-of-the-art DA methods. The experiment results demonstrate the effectiveness of the proposed framework.


# 📝  Citation
If you find our paper helpful, please give a ⭐ and cite it as follows:
```
@ARTICLE{11020658,
  author={Liang, Junye and Yang, Jiaqi and Liu, Rong and Liu, Quanwei and Zhu, Peng},
  journal={IEEE Geoscience and Remote Sensing Letters}, 
  title={Spectral Structure-Aware Initialization and Probability-Consistent Self-Training for Cross-scene Hyperspectral Image Classification}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  keywords={Feature extraction;Training;Principal component analysis;Data models;Vectors;Robustness;Hyperspectral imaging;Electronic mail;Data mining;Semantics;Hyperspectral image;Cross-scene classification;Initialization;Self-training},
  doi={10.1109/LGRS.2025.3575600}}
  ```

# 📖 Relevant Projects

[1] <strong>Dual Classification Head Self-training Network for Cross-scene Hyperspectral Image Classification, arxiv, 2025</strong> | [Paper](http://arxiv.org/abs/2502.17879)
<br><em>&ensp; &ensp; &ensp; Rong Liu, Junye Liang, Jiaqi Yang, Jiang He, Peng Zhu</em>

[2] <strong>Hyper-LKCNet: Exploring the Utilization of Large Kernel Convolution for Hyperspectral Image Classification, JSTARS, 2025</strong> | [Paper](https://ieeexplore.ieee.org/abstract/document/11007459) | [Code](https://github.com/liurongwhm/Hyper-LKNet)
<br><em>&ensp; &ensp; &ensp; Rong Liu, Zhilin Li, Jiaqi Yang , Jian Sun, and Quanwei Liu</em>

# 🔩 Requirements
CUDA Version: 12.2
Python: 3.9
torch: 2.1.0

# 📚 Dataset
The dataset directory should look like this:
datasets
├── Houston
│   ├── Houston13.mat
│   ├── Houston13_7gt.mat
│   ├── Houston18.mat
│   └── Houston18_7gt.mat
├── Pavia
│   ├── paviaU.mat
│   └── paviaU_7gt.mat
│   ├── paviaC.mat
│   └── paviaC_7gt.mat
└──  HyRANK
    ├── Dioni.mat
    └── Dioni_gt_out68.mat
    ├── Loukia.mat
    └── Loukia_gt_out68.mat

# 🔨Usage
1. Download and prepare your hyperspectral dataset. You can download Houston dataset in *./dataset/Houston*. 
2. Please change the source_name and target_name in train.py.
3. Run python train.py.
4. Default results directory is: *.*/*results*. You can check your classification maps here.


# 🎺 Statement
For any other questions please contact Junye Liang at [sysu.edu.cn](liangjy225@mail2.sysu.edu.cn).



