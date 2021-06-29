
from torch import nn
import torch
import numpy as np
import torch.nn.functional as F
import torchvision

from pathlib import Path
import matplotlib.pyplot as plt
import skimage
import os
from tqdm import tqdm
import shutil
# metric
from sklearn import metrics
# import eval_from_buster as ebust
import utils

def add_torch_overlay(im, mask_s, mask_f=None, inv=True, clone=True):
    if clone:
        im = im.clone()
    if len(im.shape) < 4:
        im = im.unsqueeze(0)
    if len(mask_s.shape) < 4:
        mask_s = mask_s.unsqueeze(0)
    if mask_f is not None and len(mask_f.shape) < 4:
        mask_f = mask_f.unsqueeze(0)
    if inv:
        mean = torch.tensor([0.485, 0.456, 0.406],
                            device=im.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225],
                           device=im.device).view(1, 3, 1, 1)
        im = im * std + mean
    for i in range(im.shape[0]):
        if mask_f is None:
            im[i, 0] += mask_s[i, 0]
        else:
            im[i, 0] += mask_f[i, 0]
            im[i, 1] += mask_s[i, 0]
    im = im.clamp_max(1.)
    im = im.clamp_min(0)
    return im

def rev_inv(im):
    mean = torch.tensor([0.485, 0.456, 0.406],
                    device=im.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225],
                        device=im.device).view(3, 1, 1)
    im = im * std + mean
    return im


@torch.no_grad()
def test_mono(data_loader, model, args, iteration, device,
              logger=None, num=None, plot=False):
    model.eval()

    metric = utils.MMetric("mask : ")

    if plot:
        fldr_name = f"fig"
        if os.path.exists(fldr_name):
            shutil.rmtree(fldr_name)
        Path(fldr_name).mkdir(exist_ok=True, parents=True)

    for i, ret in enumerate(data_loader):

        X, Y = ret
        X, Y = X.to(device), Y.to(device)

        y_det = np.zeros(Y.shape[0])
        for j in range(Y.shape[0]):
            if torch.any(Y[j] > 0.5):
                y_det[j] = 1

        pred, pred_det = model(X)
        pred = torch.sigmoid(pred)

        print(f'{i}:')

        # Source
        gt_mask_s = Y.data.cpu().numpy()
        pred_mask_s = pred.data.cpu().numpy()

        f_gt = gt_mask_s
        f_pred = pred_mask_s

        metric.update((f_gt > 0.9), (f_pred > args.thres))

        if plot:
            for ind_wrs in range(X.shape[0]):

                x_gt = add_torch_overlay(X[ind_wrs], Y[ind_wrs])
                x_pred = add_torch_overlay(X[ind_wrs],
                                        pred[ind_wrs])
                torchvision.utils.save_image(
                    rev_inv(X[ind_wrs]), f"{fldr_name}/{i}_{ind_wrs:04d}_a.png")
                torchvision.utils.save_image(
                    x_gt, f"{fldr_name}/{i}_{ind_wrs:04d}_gt.png")
                torchvision.utils.save_image(
                    x_pred, f"{fldr_name}/{i}_{ind_wrs:04d}_pred.png")

        if num is not None and i >= num:
            break
    print(f"{iteration}")
    ts = metric.final()

    return ts
  
@torch.no_grad()
def test(data_loader, model, args, iteration, device,
         logger=None, num=None, plot=False):
    
    if args.out_channel == 1:
        return test_mono(data_loader, model, args, iteration, device,
                         logger=logger, num=num, plot=plot)
        

    model.eval()

    if plot:
        fldr_name = f"fig"
        if os.path.exists(fldr_name):
            shutil.rmtree(fldr_name)
        Path(fldr_name).mkdir(exist_ok=True, parents=True)

    metric = utils.Metric()
    for i, ret in enumerate(data_loader):
        X, Y = ret
        X, Y = X.to(device), Y.to(device)
        Ys = Y[:, [1]]
        Yf = Y[:, [0]]

        pred, pred_det = model(X)
        pred = torch.softmax(pred, dim=-3)

        preds = pred[:, [1]]
        predf = pred[:, [0]]

        print(f'{i}:')
        metric.update(Y.data.cpu().numpy(), pred.data.cpu().numpy())

        # accuracy
        gt_ind =  np.argmax(Y.data.cpu().numpy(), axis=-3)
        pred_ind = np.argmax(pred.data.cpu().numpy(), axis=-3)
        tt = [np.sum(gt_ind==pred_ind), gt_ind.size]
        print()

        if logger is not None:
            ind_wrs = np.random.choice(X.shape[0], size=2)
            def fn_norm(x): return (x-x.min()) / (x.max()-x.min()+1e-8)
            x_gt = add_torch_overlay(X[ind_wrs], Ys[ind_wrs], Yf[ind_wrs])
            logger.add_images(f"Im_{i}/gt",
                              x_gt, iteration)
            x_pred = add_torch_overlay(X[ind_wrs],
                                       preds[ind_wrs],
                                       predf[ind_wrs])
            logger.add_images(f"Im_{i}/pred",
                              x_pred, iteration)

            if plot:
                torchvision.utils.save_image(
                    rev_inv(X[ind_wrs[0]]), f"{fldr_name}/{i}_a.png")
                torchvision.utils.save_image(
                    x_gt[0], f"{fldr_name}/{i}_gt.png")
                torchvision.utils.save_image(
                    x_pred[0], f"{fldr_name}/{i}_pred.png")
        if num is not None and i >= num:
            break
    print(f"{iteration}")

    out = metric.final()    
    return out
