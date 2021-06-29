"""main file for training bubblenet type comparison patch matching
"""

import os
import numpy as np
import torch
from torch import nn
from torchvision import transforms
from tensorboardX import SummaryWriter
from tqdm import tqdm
import cv2
from matplotlib import pyplot as plt
from pathlib import Path
import torch.nn.functional as F
import skimage

# custom module
import config
import models
import utils
from pathlib import Path


def add_overlay(im, mask_s):
    im = im.copy()
    im = skimage.img_as_float(im)
    im[..., 0] += mask_s[..., 0]
    im[..., 1] += mask_s[..., 1]
    im = im.clip(max=1., min=0)
    return im

if __name__ == "__main__":
    # device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    args = config.config_USC()
    args.size = tuple(int(i) for i in args.size.split("x"))

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # model name

    model = models.DOA(out_channel=args.out_channel)
    model.to(device)

    transform = utils.CustomTransform(size=args.size)

    if args.ckpt is not None:
        checkpoint = torch.load(args.ckpt)
        model.load_state_dict(checkpoint["model_state"], strict=True)

    model.eval()

    save_dir = Path("fig_test_folder")
    orig_folder_im = Path("images")
    save_dir.mkdir(exist_ok=True)

    num = None

    for i_cnt, each_f in enumerate(sorted(orig_folder_im.iterdir())):

        name = each_f.name
        im = cv2.resize(skimage.io.imread(str(each_f))[..., :3], args.size, interpolation=1)

        im = skimage.img_as_float(im)

        x, _ = transform(im)

        with torch.no_grad():
            pred, _ = model(x.unsqueeze(0).to(device))
            if args.out_channel == 3:
                pred = torch.softmax(pred, -3)
            else:
                pred = pred.sigmoid()
            pred = pred.permute(0, 2, 3, 1)
            pred = pred.squeeze()
        pred = pred.data.cpu().numpy()
        # im_pred = add_overlay(im, pred)
        im_pred = pred

        # im_pred = (im_pred - im_pred.min()) / (im_pred.max() - im_pred.min() + 1e-4)
        skimage.io.imsave(
            str(save_dir / (f"{i_cnt}_orig.png")
                ), skimage.img_as_ubyte(im)
        )
        skimage.io.imsave(
            str(save_dir / (f"{i_cnt}_mask.png")
                ), skimage.img_as_ubyte(im_pred)
        )
        print(i_cnt)

        if num is not None and i_cnt > num:
            break
