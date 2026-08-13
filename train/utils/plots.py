# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""绘图工具(train11111new 比赛版,从 utils/plots.py 精简)。

保留 val.py 需要的 output_to_target / plot_images,删除 evolve/label 统计等绘图。
"""

import math
import os
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
from ultralytics.utils.plotting import Annotator, colors

from . import threaded
from .general import xywh2xyxy, xyxy2xywh

# Settings
RANK = int(os.getenv("RANK", "-1"))
matplotlib.rc("font", size=11)
matplotlib.use("Agg")  # for writing to files only


# 将模型输出转换为 [batch_id, class_id, x, y, w, h, conf] 格式用于绘图
def output_to_target(output, max_det=300):
    """Converts model output to [batch_id, class_id, x, y, w, h, conf] format for plotting, handling up to `max_det`
    detections.
    """
    targets = []
    for i, o in enumerate(output):
        box, conf, cls = o[:max_det, :6].cpu().split((4, 1, 1), 1)
        j = torch.full((conf.shape[0], 1), i)
        targets.append(torch.cat((j, cls, xyxy2xywh(box), conf), 1))
    return torch.cat(targets, 0).numpy()


# 把 batch 图像与标签/预测框画成网格大图并保存
@threaded
def plot_images(images, targets, paths=None, fname="images.jpg", names=None):
    """Plots a grid of images with labels, optionally resizing and annotating with target boxes and names."""
    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    max_size = 1920  # max image size
    max_subplots = 16  # max image subplots, i.e. 4x4
    bs, _, h, w = images.shape  # batch size, _, height, width
    bs = min(bs, max_subplots)  # limit plot images
    ns = np.ceil(bs**0.5)  # number of subplots (square)
    if np.max(images[0]) <= 1:
        images *= 255  # de-normalise (optional)

    # Build Image
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)  # init
    for i, im in enumerate(images):
        if i == max_subplots:  # if last batch has fewer images than we expect
            break
        x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
        im = im.transpose(1, 2, 0)
        mosaic[y : y + h, x : x + w, :] = im

    # Resize (optional)
    scale = max_size / ns / max(h, w)
    if scale < 1:
        h = math.ceil(scale * h)
        w = math.ceil(scale * w)
        mosaic = cv2.resize(mosaic, tuple(int(x * ns) for x in (w, h)))

    # Annotate
    fs = int((h + w) * ns * 0.01)  # font size
    annotator = Annotator(mosaic, line_width=round(fs / 10), font_size=fs, pil=True, example=names)
    for block_index in range(i + 1):
        x, y = int(w * (block_index // ns)), int(h * (block_index % ns))  # block origin
        annotator.rectangle([x, y, x + w, y + h], None, (255, 255, 255), width=2)  # borders
        if paths:
            annotator.text(
                [x + 5, y + 5], text=Path(paths[block_index]).name[:40], txt_color=(220, 220, 220)
            )  # filenames
        if len(targets) > 0:
            ti = targets[targets[:, 0] == block_index]  # image targets
            boxes = xywh2xyxy(ti[:, 2:6]).T
            classes = ti[:, 1].astype("int")
            labels = ti.shape[1] == 6  # labels if no conf column
            conf = None if labels else ti[:, 6]  # check for confidence presence (label vs pred)

            if boxes.shape[1]:
                if boxes.max() <= 1.01:  # if normalized with tolerance 0.01
                    boxes[[0, 2]] *= w  # scale to pixels
                    boxes[[1, 3]] *= h
                elif scale < 1:  # absolute coords need scale if image scales
                    boxes *= scale
            boxes[[0, 2]] += x
            boxes[[1, 3]] += y
            for j, box in enumerate(boxes.T.tolist()):
                cls = classes[j]
                color = colors(cls)
                cls = names[cls] if names else cls
                if labels or conf[j] > 0.25:  # 0.25 conf thresh
                    label = f"{cls}" if labels else f"{cls} {conf[j]:.1f}"
                    annotator.box_label(box, label, color=color)
    annotator.im.save(fname)  # save
