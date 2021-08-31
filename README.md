# DOA-GAN

[![Paper](https://img.shields.io/badge/paper-paper)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Islam_DOA-GAN_Dual-Order_Attentive_Generative_Adversarial_Network_for_Image_Copy-Move_Forgery_CVPR_2020_paper.pdf)

## Requirements
- PyTorch-1.4+

Install the required packages by:

```
pip install -r requirements.txt
```


## Pretrained models

Pretrained models can be downloaded from [drive link](https://drive.google.com/drive/folders/1tFol0YerCZdxEiutK_J95jkG3JvAYtoj?usp=sharing)

## Test on USC-ISI

```python
python main.py --dataset usc --ckpt ./ckpt/three_channel.pkl [--plot]
```

`--plot` flag will save the output images in `fig/` directory.

## Test on CASIA/COMO

```python
python main.py --dataset [casia/como] --ckpt ./ckpt/single_channel.pkl [--plot]
```


## Test on Custom folder

Put forged images in `images/` folder, and run

```
python run_on_folder.py --ckpt [model_weight_file] --out-channel [1 or 3]
```

e.g., 
```
python run_on_folder.py --ckpt ./ckpt/three_channel.pkl --out-channel 3
```

The output masks will be saved in `fig_test_folder` folder.

## Citation
```
@InProceedings{Islam_2020_CVPR,
author = {Islam, Ashraful and Long, Chengjiang and Basharat, Arslan and Hoogs, Anthony},
title = {DOA-GAN: Dual-Order Attentive Generative Adversarial Network for Image Copy-Move Forgery Detection and Localization},
booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```
