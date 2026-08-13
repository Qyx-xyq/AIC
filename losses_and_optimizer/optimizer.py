# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""YOLOv3 优化器配置(比赛用,自包含)。

内容原样提取自 YOLOv3 源码 ``utils/torch_utils.py`` 的 ``smart_optimizer``;
``LOGGER`` / ``colorstr`` 精简自 ``utils/general.py``(等价于 ultralytics 包中的同名对象)。

依赖: ``torch``。
"""

import logging

import torch
from torch import nn


# 日志记录器(等价于 yolov3 中从 ultralytics 复用的 LOGGER)
LOGGER = logging.getLogger("yolov3")
if not LOGGER.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    LOGGER.addHandler(_handler)
    LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False


# 给字符串添加 ANSI 颜色(源自 ultralytics.utils.colorstr)
def colorstr(*input):
    """Color a string using ANSI escape codes; e.g. colorstr('blue', 'bold', 'hello')."""
    *args, string = input if len(input) > 1 else ("blue", "bold", input[0])  # color arguments, string
    colors = {
        "black": "\033[30m",
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "bright_black": "\033[90m",
        "bright_red": "\033[91m",
        "bright_green": "\033[92m",
        "bright_yellow": "\033[93m",
        "bright_blue": "\033[94m",
        "bright_magenta": "\033[95m",
        "bright_cyan": "\033[96m",
        "bright_white": "\033[97m",
        "end": "\033[0m",
        "bold": "\033[1m",
        "underline": "\033[4m",
    }
    return "".join(colors[x] for x in args) + f"{string}" + colors["end"]


# 按参数类型分组(带衰减权重 / 不带衰减的 BN 权重 / 偏置)创建智能优化器
def smart_optimizer(model, name="Adam", lr=0.001, momentum=0.9, decay=1e-5):
    """Initializes a smart optimizer for YOLOv3 with custom parameter groups for different weight decays and biases."""
    g = [], [], []  # optimizer parameter groups
    bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
    for v in model.modules():
        for p_name, p in v.named_parameters(recurse=0):
            if p_name == "bias":  # bias (no decay)
                g[2].append(p)
            elif p_name == "weight" and isinstance(v, bn):  # weight (no decay)
                g[1].append(p)
            else:
                g[0].append(p)  # weight (with decay)

    if name == "Adam":
        optimizer = torch.optim.Adam(g[2], lr=lr, betas=(momentum, 0.999))  # adjust beta1 to momentum
    elif name == "AdamW":
        optimizer = torch.optim.AdamW(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
    elif name == "RMSProp":
        optimizer = torch.optim.RMSprop(g[2], lr=lr, momentum=momentum)
    elif name == "SGD":
        optimizer = torch.optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
    else:
        raise NotImplementedError(f"Optimizer {name} not implemented.")

    optimizer.add_param_group({"params": g[0], "weight_decay": decay})  # add g0 with weight_decay
    optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})  # add g1 (BatchNorm2d weights)
    LOGGER.info(
        f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}) with parameter groups "
        f"{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias"
    )
    return optimizer
