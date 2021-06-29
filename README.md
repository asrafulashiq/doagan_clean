# Copy-Move

## Test USC-ISI
---

```python
python -m pdb main.py --dataset usc --ckpt ./ckpt/three_channel.pkl [--plot]
```

`--plot` flag will save the output images in `fig/` directory.

## Test CASIA/COMO
---

```python
python -m pdb main.py --dataset [casia/como] --ckpt ./ckpt/three_channel.pkl [--plot]
```


## Test on Custom folder
---

put forged images in `images/` folder, and run

```
python -m pdb run_on_folder.py --ckpt [model_weight_file] --out-channel [1 or 3]
```

e.g., 
```
python -m pdb run_on_folder.py --ckpt ./ckpt/three_channel.pkl --out-channel 3
```

The output masks will be saved in `fig_test_folder` folder.