
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
from test import test
import dataset
import utils


if __name__ == "__main__":
    # device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    args = config.config_USC()

    if args.dataset == 'usc':
        # args = config.config_USC()
        args.out_channel = 3
    elif args.dataset == 'casia':
        args = config.config_CASIA()
        args.out_channel = 1
    elif args.dataset == 'como':
        args = config.config_COMO()
        args.out_channel = 1

    args.size = tuple(int(i) for i in args.size.split("x"))

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # model name
    model_name = args.model + "_" + \
        args.dataset + args.suffix

    print(f"Model Name: {model_name}")

    # model
    model = models.DOA(out_channel=args.out_channel)
    model.to(device)

    if args.dataset == 'usc':
        data = dataset.USCISI_CMD_Dataset(lmdb_dir=args.lmdb_dir, args=args,
                                                 sample_file=args.train_key)
    elif args.dataset == 'casia':
        data = dataset.Dataset_CASIA(args)
    elif args.dataset == 'como':
        data = dataset.Dataset_COMO(args)

    test_data_loader = torch.utils.data.DataLoader(
        data, batch_size=args.batch_size, shuffle=True, num_workers=2
    )

    if args.ckpt is not None:
        checkpoint = torch.load(args.ckpt)
        model.load_state_dict(checkpoint["model_state"], strict=True)

    iteration = 0
    print(f"test on {args.dataset}")
    test(test_data_loader, model, args, iteration, device,
        logger=None, num=30, plot=args.plot)
