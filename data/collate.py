# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Batch 打包函数:把多张图叠成张量、把标签拼接成统一张量。

原样提取自 ``utils/dataloaders.py`` 中 ``LoadImagesAndLabels.collate_fn`` /
``collate_fn4`` 两个静态方法,改为模块级函数(逻辑不变)。
"""

import random

import torch
import torch.nn.functional as F


# 把 batch 打包成 (堆叠图像张量, 拼接标签张量, path, shapes),并给每个标签写入所属图像索引(供 build_targets 用)。
def collate_fn(batch):
    """Collates batch of images, labels, paths, and shapes, indexing labels for target image identification."""
    im, label, path, shapes = zip(*batch)  # transposed
    for i, lb in enumerate(label):
        lb[:, 0] = i  # add target image index for build_targets()
    return torch.stack(im, 0), torch.cat(label, 0), path, shapes


# 三模态:每个样本返回 (visible, infared, depth, label, path, shapes),分别堆叠三个模态图像。
def collate_fn3(batch):
    """Collates tri-modal batches into three stacked image tensors plus concatenated labels, paths and shapes."""
    im0, im1, im2, label, path, shapes = zip(*batch)  # transposed
    for i, lb in enumerate(label):
        lb[:, 0] = i  # add target image index for build_targets()
    return (
        torch.stack(im0, 0),
        torch.stack(im1, 0),
        torch.stack(im2, 0),
        torch.cat(label, 0),
        path,
        shapes,
    )


# quad 模式:每 4 张拼成 1 张大图(2×2)或 2 倍上采样,再打包成 batch。
def collate_fn4(batch):
    """Batches images, labels, paths, and shapes by grouping every 4 items for dataset loading."""
    im, label, path, shapes = zip(*batch)  # transposed
    n = len(shapes) // 4
    im4, label4, path4, shapes4 = [], [], path[:n], shapes[:n]

    ho = torch.tensor([[0.0, 0, 0, 1, 0, 0]])
    wo = torch.tensor([[0.0, 0, 1, 0, 0, 0]])
    s = torch.tensor([[1, 1, 0.5, 0.5, 0.5, 0.5]])  # scale
    for i in range(n):  # zidane torch.zeros(16,3,720,1280)  # BCHW
        i *= 4
        if random.random() < 0.5:
            im1 = F.interpolate(im[i].unsqueeze(0).float(), scale_factor=2.0, mode="bilinear", align_corners=False)[
                0
            ].type(im[i].type())
            lb = label[i]
        else:
            im1 = torch.cat((torch.cat((im[i], im[i + 1]), 1), torch.cat((im[i + 2], im[i + 3]), 1)), 2)
            lb = torch.cat((label[i], label[i + 1] + ho, label[i + 2] + wo, label[i + 3] + ho + wo), 0) * s
        im4.append(im1)
        label4.append(lb)

    for i, lb in enumerate(label4):
        lb[:, 0] = i  # add target image index for build_targets()

    return torch.stack(im4, 0), torch.cat(label4, 0), path4, shapes4
