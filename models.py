import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models
from torchvision import transforms
import numpy as np
from matplotlib import pyplot as plt
import skimage
import cv2


def get_topk(x, k=10, dim=-3):
    # b, c, h, w = x.shape
    val, _ = torch.topk(x, k=k, dim=dim)
    return val


class ZeroWindow:
    def __init__(self):
        self.store = {}

    def __call__(self, x_in, h, w, rat_s=0.1):
        sigma = h * rat_s, w * rat_s
        # c = h * w
        b, c, h2, w2 = x_in.shape
        key = str(x_in.shape) + str(rat_s)
        if key not in self.store:
            ind_r = torch.arange(h2).float()
            ind_c = torch.arange(w2).float()
            ind_r = ind_r.view(1, 1, -1, 1).expand_as(x_in)
            ind_c = ind_c.view(1, 1, 1, -1).expand_as(x_in)

            # center
            c_indices = torch.from_numpy(np.indices((h, w))).float()
            c_ind_r = c_indices[0].reshape(-1)
            c_ind_c = c_indices[1].reshape(-1)

            cent_r = c_ind_r.reshape(1, c, 1, 1).expand_as(x_in)
            cent_c = c_ind_c.reshape(1, c, 1, 1).expand_as(x_in)

            def fn_gauss(x, u, s):
                return torch.exp(-(x - u) ** 2 / (2 * s ** 2))

            gaus_r = fn_gauss(ind_r, cent_r, sigma[0])
            gaus_c = fn_gauss(ind_c, cent_c, sigma[1])
            out_g = 1 - gaus_r * gaus_c
            out_g = out_g.to(x_in.device)
            self.store[key] = out_g
        else:
            out_g = self.store[key]
        out = out_g * x_in
        return out


class Corr(nn.Module):
    def __init__(self, topk=3):
        super().__init__()
        # self.h = hw[0]
        # self.w = hw[1]
        self.topk = topk
        self.zero_window = ZeroWindow()

        self.alpha = nn.Parameter(torch.tensor(
            5., dtype=torch.float32))

    def forward(self, x):
        b, c, h1, w1 = x.shape
        h2 = h1
        w2 = w1

        xn = F.normalize(x, p=2, dim=-3)
        x_aff_o = torch.matmul(xn.permute(0, 2, 3, 1).view(b, -1, c),
                               xn.view(b, c, -1))  # h1 * w1, h2 * w2

        # zero out same area corr
        # x_aff = _zero_window(x_aff_o.view(b, -1, h1, w1), h1, w1, rat_s=0.05).reshape(b, h1*w1, h2*w2)
        x_aff = self.zero_window(x_aff_o.view(b, -1, h1, w1), h1, w1, rat_s=0.05).reshape(b, h1*w1, h2*w2)

        x_c = F.softmax(x_aff * self.alpha, dim=-1) * \
            F.softmax(x_aff * self.alpha, dim=-2)
        x_c = x_c.reshape(b, h1, w1, h2, w2)

        xc_o = x_c.view(b, h1*w1, h2, w2)
        val = get_topk(xc_o, k=self.topk, dim=-3)

        x_soft = xc_o / (xc_o.sum(dim=-3, keepdim=True) + 1e-8)
        return val, x_soft


def std_mean(x):
    return (x-x.mean(dim=-3, keepdim=True))/(1e-8+x.std(dim=-3, keepdim=True))


def plot_mat(x, name, cmap='jet', size=(120, 120)):
    x = x.data.cpu().numpy()
    xr = cv2.resize(x, size, interpolation=cv2.INTER_LINEAR)

    plt.matshow(xr, cmap=cmap)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f'{name}.png', bbox_inches='tight', pad_inches=0,
                frameon=False, transparent=True)


def plot_plt(x, name='1', size=(240, 240), cmap='jet'):
    if x.shape[0] == 2:
        x = torch.cat((x, x[[0]]), dim=-3)
    x = F.interpolate(x.unsqueeze(0), size=size,
                      mode='bilinear').squeeze(0)

    def fn(x): return (x-x.min())/(x.max()-x.min()+1e-8)
    xn = fn(x)
    xn = x.permute(1, 2, 0).data.cpu().numpy()[..., 0]
    plt.imsave(f'{name}_m.png', xn, cmap=cmap)


def plot(x, name='1', size=(240, 240)):
    if x.shape[0] == 2:
        x = torch.cat((x, x[[0]]), dim=-3)
    x = F.interpolate(x.unsqueeze(0), size=size,
                      mode='bilinear').squeeze(0)

    def fn(x): return (x-x.min())/(x.max()-x.min()+1e-8)
    torchvision.utils.save_image(fn(x), f'{name}.png')


class DetectionBranch(nn.Module):
    def __init__(self, in_cat=896):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_cat, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.lin = nn.Sequential(
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1))

        self.apply(weights_init_normal)

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.conv1(x)

        x_cat = torch.cat((
            F.adaptive_avg_pool2d(x, (1, 1)),
            F.adaptive_max_pool2d(x, (1, 1))
        ), dim=-3)

        x_cat = x_cat.reshape(b, 128)
        y = self.lin(x_cat)
        return y


class DOA(nn.Module):
    def __init__(self, out_channel=3, hw=(40, 40), topk=20):
        super().__init__()
        self.hw = hw
        self.topk = topk

        self.encoder = Extractor_VGG19()

        self.corrLayer = Corr(topk=topk)

        in_cat = 896

        self.val_conv = nn.Sequential(
            nn.Conv2d(topk, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )

        self.aspp_mask = models.segmentation.deeplabv3.ASPP(in_channels=in_cat,
                                                            atrous_rates=[12, 24, 36])

        self.head_mask = nn.Sequential(
            nn.Conv2d(4 * 256+topk, 2 * 256, 1),
            nn.BatchNorm2d(2 * 256),
            nn.ReLU(),
            nn.Conv2d(2 * 256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, out_channel, 1)
        )

        self.aspp_forge = models.segmentation.deeplabv3.ASPP(in_channels=in_cat,
                                                             atrous_rates=[6, 12, 24])

        # detection branch
        self.detection = DetectionBranch(4*256+topk)

        self.head_mask.apply(weights_init_normal)

        self.val_conv.apply(weights_init_normal)

        # self.dropout = lambda x: x
        self.dropout = nn.Dropout2d(p=0.6)

    def forward(self, x):
        input = x
        b, c, h, w = x.shape
        x = self.encoder(x, out_size=self.hw)

        val, ind = self.corrLayer(x)

        # attention weight
        val_conv = self.val_conv(val)

        #### Mask part : M  ####
        x_as = self.dropout(self.aspp_mask(x)) * val_conv

        #### Forge Part: F ####
        x_asf = self.dropout(self.aspp_forge(x)) * val_conv

        x_as_p = self.dropout(self.non_local(x_as, ind))
        x_asf_p = self.dropout(self.non_local(x_asf, ind))

        # Final Mask
        x_cat = torch.cat((x_as, x_as_p, x_asf, x_asf_p, val), dim=-3)
        out1 = self.head_mask(x_cat)

        # final detection
        out_det = self.detection(x_cat)

        out = F.interpolate(out1, size=(h, w), mode='bilinear',
                            align_corners=True)

        return out, out_det

    def non_local(self, x, ind):
        b, c, h2, w2 = x.shape
        b, _, h1, w1 = ind.shape

        x = x.reshape(b, c, -1)
        ind = ind.reshape(b, h2 * w2, h1 * w1)
        out = torch.bmm(x, ind).reshape(b, c, h1, w1)
        return out

    def set_bn_to_eval(self):
        def fn(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                m.eval()
        self.apply(fn)



def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 or classname.find("Linear") != -1:
        torch.nn.init.kaiming_normal_(m.weight.data)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Extractor_VGG19(nn.Module):
    def __init__(self):
        super().__init__()
        cnn_temp = torchvision.models.vgg19(pretrained=True).features

        self.layer1 = nn.Sequential(cnn_temp[:10])  # stride 4
        self.layer2 = nn.Sequential(cnn_temp[10:19])  # s 8
        self.layer3 = nn.Sequential(cnn_temp[19:28])  # s 16

    def forward(self, x, out_size=(40, 40)):
        x1 = self.layer1(x)
        x1_u = F.interpolate(
            x1, size=out_size, mode='bilinear', align_corners=True)

        x2 = self.layer2(x1)
        x2_u = F.interpolate(
            x2, size=out_size, mode='bilinear', align_corners=True)

        x3 = self.layer3(x2)
        x3_u = F.interpolate(
            x3, size=out_size, mode='bilinear', align_corners=True)
        # return x3
        x_all = torch.cat([x1_u, x2_u, x3_u], dim=-3)

        return x_all
