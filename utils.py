from sklearn.metrics import precision_recall_fscore_support
from scipy.ndimage import morphology
from torch import nn
from torch.nn import functional as F
import torch
from torchvision import transforms
import numpy as np
import cv2
import skimage
from skimage import io
from skimage import transform

from scipy.ndimage import morphology
from scipy.ndimage import measurements

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage



def fscore(T_score):
    Fp, Fn, Tp = T_score[1:]
    f_score = 2 * Tp / (2 * Tp + Fp + Fn)
    return f_score

def precision(T_score):
    Fp, _, Tp = T_score[1:]
    recall = Tp / (Tp + Fp + 1e-8)
    return recall


def recall(T_score):
    _, Fn, Tp = T_score[1:]
    recall = Tp / (Tp + Fn + 1e-8)
    return recall 


def conf_mat(labels, preds):
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels).ravel()
    if not isinstance(preds, np.ndarray):
        preds = np.array(preds).ravel()

    tp = np.sum(preds[labels == 1] == 1)
    tn = np.sum(preds[labels == 0] == 0)
    fp = np.sum(preds[labels == 0] == 1)
    fn = np.sum(preds[labels == 1] == 0)
    return np.array([tn, fp, fn, tp])


class SimTransform():
    def __init__(self, size=(224, 224)):
        if isinstance(size, int) or isinstance(size, float):
            size = (size, size)
        # scale
        self.scale = np.random.choice(
            np.linspace(0.9, 1.1, 30)
        )
        # rotation
        self.rot = np.random.choice(
            np.linspace(-np.pi/10, np.pi/10, 50)
        )
        # translate
        self.translate = (
            np.random.choice(np.linspace(-0.05 * size[1], 0.1 * size[1], 50)),
            np.random.choice(np.linspace(-0.05 * size[0], 0.1 * size[0], 50))
        )

        # flip lr
        self.flip = np.random.rand() > 0.5

        self.tfm = transform.SimilarityTransform(
            scale=self.scale,
            translation=self.translate)

    def __call__(self, im=None, mask=None):
        if im is not None:
            im = transform.warp(im, self.tfm)
            # im = transform.rotate(im, self.rot)
            if self.flip:
                im = np.flip(im, 1).copy()
        if mask is not None:
            mask = transform.warp(mask, self.tfm)
            # mask = transform.rotate(mask, self.rot)
            if self.flip:
                mask = np.flip(mask, 1).copy()
            mask[mask > 0.5] = 1
            mask[mask <= 0.5] = 0
        return im, mask


class CustomTransform:
    def __init__(self, size=224):
        if isinstance(size, int) or isinstance(size, float):
            self.size = (size, size)
        else:
            self.size = size
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        # self.normalize = transforms.Normalize(mean, std)
        self.to_tensor = transforms.ToTensor()

    def resize(self, img=None, mask=None):
        if img is not None:
            img = skimage.img_as_float32(img)
            if img.shape[0] != self.size[0] or img.shape[1] != self.size[1]:
                img = cv2.resize(
                    img, self.size, interpolation=cv2.INTER_LINEAR)

        if mask is not None:
            if mask.shape[0] != self.size[0] or mask.shape[1] != self.size[1]:
                mask = cv2.resize(
                    mask, self.size, interpolation=cv2.INTER_NEAREST)
        return img, mask

    def inverse(self, x, mask=False):
        if x.is_cuda:
            x = x.squeeze().data.cpu().numpy()
        else:
            x = x.squeeze().data.numpy()
        x = x.transpose((1, 2, 0))
        if not mask:
            x = x * self.std + self.mean
        return x

    def __call__(self, img=None, mask=None, other_tfm=None):
        img, mask = self.resize(img, mask)
        if other_tfm is not None:
            img, mask = other_tfm(img, mask)
        if img is not None:
            img = (img - self.mean) / self.std
            img = self.to_tensor(img).float()

        if mask is not None:
            mask = self.to_tensor(mask).float()

        return img, mask


def custom_transform_images(images=None, masks=None, size=224, other_tfm=None):
    tsfm = CustomTransform(size=size)
    X, Y = None, None
    if images is not None:
        X = torch.zeros((images.shape[0], 3, size, size), dtype=torch.float32)
        for i in range(images.shape[0]):
            X[i] = tsfm(img=images[i], other_tfm=other_tfm)
    if masks is not None:
        Y = torch.zeros((masks.shape[0], 1, size, size), dtype=torch.float32)
        for i in range(masks.shape[0]):
            _, Y[i, 0] = tsfm(img=None, mask=masks[i], other_tfm=other_tfm)

    return X, Y


def add_overlay(im, m1, m2=None, alpha=0.5, c1=[0, 1, 0], c2=[1, 0, 0]):
    r, c = im.shape[:2]

    M1 = np.zeros((r, c, 3), dtype=np.float32)
    M2 = np.zeros((r, c, 3), dtype=np.float32)

    if m2 is not None:
        M1[m1 > 0] = c1
        M2[m2 > 0] = c2
        M = cv2.addWeighted(M1, alpha, M2, 1 - alpha, 0, None)
    else:
        M1[m1 > 0] = c1
        M = M1

    I = cv2.addWeighted(im, alpha, M, 1 - alpha, 0, None)

    return I


class MultiPagePdf:
    def __init__(self, total_im, out_name, nrows=4, ncols=4, figsize=(8, 6)):
        """init

        Keyword Arguments:
            total_im {int} -- #images
            nrows {int} -- #rows per page (default: {4})
            ncols {int} -- #columns per page (default: {4})
            figsize {tuple} -- fig size (default: {(8, 6)})
        """
        self.total_im = total_im
        self.nrows = nrows
        self.ncols = ncols
        self.figsize = tuple(figsize)
        self.out_name = out_name

        # create figure and axes
        total_pages = int(np.ceil(total_im / (nrows * ncols)))

        self.figs = []
        self.axes = []

        for _ in range(total_pages):
            f, a = plt.subplots(nrows, ncols)

            f.set_size_inches(figsize)
            self.figs.append(f)
            self.axes.extend(a.flatten())

        self.cnt_ax = 0

    def plot_one(self, x, *args, **kwargs):
        ax = self.axes[self.cnt_ax]
        ax.imshow(x, *args, **kwargs)  # prediction
        # ax.imshow(x[0])  # ground truth

        ax.set_xticks([])
        ax.set_yticks([])

        self.cnt_ax += 1
        return ax

    def final(self):
        with PdfPages(self.out_name) as pdf:
            for fig in self.figs:
                fig.tight_layout()
                pdf.savefig(fig)
        plt.close("all")


class MMetric():
    def __init__(self, name=""):
        self.T = np.zeros(4)
        self.fscore = []
        self.prec = []
        self.rec = []
        self.name = name

    def update(self, gt, pred, batch_mode=True, log=True):

        if batch_mode:
            num = gt.shape[0]
            arr = []
            for i in range(num):
                fs = self._update(gt[i], pred[i], log=False)
                if fs != -1:
                    arr.append(fs)
            if len(arr) > 0:
                print(f"\t\t{self.name} f score : {np.mean(arr):.4f}")
        else:
            self._update(gt, pred, log=log)

    def _update(self, gt, pred, log=True):
        gt = gt.ravel()
        pred = pred.ravel()

        tt = conf_mat(gt, pred)
        self.T += tt

        if np.all(gt == 0):
            return -1

        prec = precision(tt)
        rec = recall(tt)
        fs = fscore(tt)

        self.prec.append(prec)
        self.rec.append(rec)
        self.fscore.append(fs)

        if log:
            print(
                f"{self.name} precision : {prec:.4f}, recall : {rec:.4f}, f1 : {fs:.4f}")
        return fs

    def final(self):
        # protocal A
        print(f"\n{self.name} ")
        print("-"*50)
        # print("\nProtocol A:")
        # print(
        #     f"precision : {precision(self.T):.4f}, recall : {recall(self.T):.4f}, f1 : {fscore(self.T):.4f}")

        # protocol B
        print("\nProtocol B:")
        print(
            f"precision : {np.mean(self.prec):.4f}, recall : {np.mean(self.rec):.4f}, f1 : {np.mean(self.fscore):.4f}")

        return np.mean(self.fscore)


class Metric():
    def __init__(self, dims=3, names=["forge", "source", "pristine"]):
        self.names = names
        self.dims = dims
        assert len(names) == dims

        self.list_metrics = []
        for i in range(dims):
            self.list_metrics.append(MMetric(name=names[i]))

    def update(self, gt, pred, batch_mode=True):
        ind_gt = np.argmax(gt, axis=-3)
        ind_pred = np.argmax(pred, axis=-3)
        for i in range(self.dims):
            self.list_metrics[i].update(
                ind_gt == i, ind_pred == i, batch_mode=batch_mode
            )

    def final(self):
        sc = []
        for i in range(self.dims):
            sc.append(self.list_metrics[i].final())
        return np.mean(sc)

import collections

class Metric_image(object):
    def __init__(self):
        self.gt = []
        self.pred = []
    
    def update(self, _gt, _pred, thres=0.5):
        _gt = _gt > thres
        _pred = _pred > thres

        if isinstance(_gt, collections.abc.Iterable):
            self.gt.extend(list(_gt))
            self.pred.extend(list(_pred))
        else:
            self.gt.append(_gt)
            self.pred.append(_pred)

    def final(self):
        pr, re, f, _ = precision_recall_fscore_support(self.gt, self.pred, average='binary')

        print("Image level score")
        print(f"precision: {pr:.4f}, recall: {re:.4f}, f-score: {f :.4f} ")


class ImAug():
    def __init__(self):
        sometimes = lambda aug: iaa.Sometimes(0.50, aug)

        self.seq_for_coco_back = iaa.Sequential([
            sometimes(iaa.Affine(rotate=(-45, 45))),  # rotate by -45 to 45 degrees (affects segmaps)
            # sometimes(iaa.ElasticTransformation(alpha=50, sigma=5)),  # apply water effect (affects segmaps)
            # sometimes(iaa.Fliplr(0.10)), # horizontally flip 10% of the images
        ], random_order=True)

    @staticmethod
    def apply_contrast(im):
        seq = iaa.ContrastNormalization((0.75, 1.5))
        im = seq(images = skimage.img_as_ubyte(im))
        im = skimage.img_as_float32(im)
        return im

    @staticmethod
    def apply_water(mask):
        seq = iaa.ElasticTransformation(alpha=50, sigma=5)
        segmap = SegmentationMapsOnImage(skimage.img_as_ubyte(mask)[..., None], shape=mask.shape)
        mask = seq(segmentation_maps = segmap)
        mask = skimage.img_as_float32(mask.get_arr().squeeze())
        return mask

    def apply_coco_back(self, im, mask):
        im = skimage.img_as_ubyte(im)
        mask = skimage.img_as_ubyte(mask)[..., None]
        segmap = SegmentationMapsOnImage(mask, shape=im.shape)
        images_aug_i, segmaps_aug_i = self.seq_for_coco_back(image=im, segmentation_maps=segmap)
        im = skimage.img_as_float32(images_aug_i)
        mask = skimage.img_as_float32(segmaps_aug_i.get_arr().squeeze())
        return im, mask



def patch_transform(im_mask, mask_bb, new_centroid, translate=None, scale=None):
    if len(mask_bb) < 4:
        return im_mask.copy()

    patch_mask = im_mask[mask_bb[1]:mask_bb[3], mask_bb[0]:mask_bb[2]]
    resized_patch = cv2.resize(patch_mask, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_NEAREST)

    hp, wp = resized_patch.shape[:2]

    topx = int(max(0, new_centroid[0] - wp/2))
    topy = int(max(0, new_centroid[1] - hp/2))

    bottomx = int(min(topx+wp, im_mask.shape[1]))
    bottomy = int(min(topy+hp, im_mask.shape[0]))

    w, h = bottomx - topx, bottomy - topy

    new_mask = np.zeros(im_mask.shape, dtype=patch_mask.dtype)

    new_mask[topy:topy+h, topx:topx+w] = resized_patch[:h, :w]

    return new_mask


def splice(img_target, img_source, img_mask):

    if img_target.shape != img_source.shape:
        img_target = skimage.transform.resize(
            img_target, img_mask.shape[:2], anti_aliasing=False, mode='reflect'
        )
        # img_target = skimage.img_as_ubyte(img_target)

    if len(img_mask.shape) < 3:
        img_mask = img_mask[..., None]

    img_mask = (img_mask > 0)
    if img_mask.dtype != np.float32:
        img_mask = img_mask.astype(np.float32)
    img_mani = img_mask * img_source + img_target * (1 - img_mask)
    # img_mani = img_mani.astype(np.uint8)

    return img_mani

