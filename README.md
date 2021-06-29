# DOA-GAN

[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Islam_DOA-GAN_Dual-Order_Attentive_Generative_Adversarial_Network_for_Image_Copy-Move_Forgery_CVPR_2020_paper.pdf)

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
python -m pdb main.py --dataset usc --ckpt ./ckpt/three_channel.pkl [--plot]
```

`--plot` flag will save the output images in `fig/` directory.

## Test on CASIA/COMO

```python
python -m pdb main.py --dataset [casia/como] --ckpt ./ckpt/single_channel.pkl [--plot]
```


## Test on Custom folder

Put forged images in `images/` folder, and run

```
python -m pdb run_on_folder.py --ckpt [model_weight_file] --out-channel [1 or 3]
```

e.g., 
```
python -m pdb run_on_folder.py --ckpt ./ckpt/three_channel.pkl --out-channel 3
```

The output masks will be saved in `fig_test_folder` folder.