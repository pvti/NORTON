from typing import Any, Callable, List, Optional
import torch
import torch.nn as nn

from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models._utils import _ovewrite_value_param
from torchvision.models.detection.backbone_utils import (
    _validate_trainable_layers,
    BackboneWithFPN,
)
from torchvision.ops.feature_pyramid_network import ExtraFPNBlock, LastLevelMaxPool

import sys
sys.path.append('..')
from models.imagenet import resnet_50, ResNet50


def fasterrcnn_CPresnet50_fpn(
    *,
    weights=None,
    num_classes: Optional[int] = None,
    weights_backbone=None,
    compress_rate=[0.0] * 53,
    rank=0,
    trainable_backbone_layers: Optional[int] = None,
    **kwargs: Any,
) -> FasterRCNN:
    """
    Simplified from https://github.com/pytorch/vision/blob/main/torchvision/models/detection/faster_rcnn.py

    Faster R-CNN model with a ResNet-50-FPN backbone from the `Faster R-CNN: Towards Real-Time Object
    Detection with Region Proposal Networks <https://arxiv.org/abs/1506.01497>`__
    paper.

    .. betastatus:: detection module

    Args:
        weights (:class:`~torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        num_classes (int, optional): number of output classes of the model (including the background)
        weights_backbone (:class:`~torchvision.models.ResNet50_Weights`, optional): The
            pretrained weights for the backbone.
        trainable_backbone_layers (int, optional): number of trainable (not frozen) layers starting from
            final block. Valid values are between 0 and 5, with 5 meaning all backbone layers are
            trainable. If ``None`` is passed (the default) this value is set to 3.
        **kwargs: parameters passed to the ``torchvision.models.detection.faster_rcnn.FasterRCNN``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/detection/faster_rcnn.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights
        :members:
    """

    if weights is not None:
        weights_backbone = None
        num_classes = _ovewrite_value_param(
            "num_classes", num_classes, len(weights.meta["categories"])
        )
    elif num_classes is None:
        num_classes = 91

    is_trained = weights is not None or weights_backbone is not None
    trainable_backbone_layers = _validate_trainable_layers(
        is_trained, trainable_backbone_layers, 5, 3
    )

    backbone = resnet_50(compress_rate=compress_rate, rank=rank)
    if weights_backbone is not None:
        ckpt = torch.load(weights_backbone)
        backbone.load_state_dict(ckpt['state_dict'])
    backbone = _resnet_fpn_extractor(backbone, trainable_backbone_layers)
    print(backbone)

    model = FasterRCNN(backbone, num_classes=num_classes, **kwargs)

    if weights is not None:
        ckpt = torch.load(weights)
        model.load_state_dict(ckpt['model'])

    return model


def _resnet_fpn_extractor(
    backbone: ResNet50,
    trainable_layers: int,
    returned_layers: Optional[List[int]] = None,
    extra_blocks: Optional[ExtraFPNBlock] = None,
    norm_layer: Optional[Callable[..., nn.Module]] = None,
) -> BackboneWithFPN:
    """
    Modified from https://github.com/pytorch/vision/blob/main/torchvision/models/detection/backbone_utils.py
    """

    # select layers that won't be frozen
    if trainable_layers < 0 or trainable_layers > 5:
        raise ValueError(
            f"Trainable layers should be in the range [0,5], got {trainable_layers}"
        )
    layers_to_train = ["layer4", "layer3", "layer2", "layer1", "conv1"][
        :trainable_layers
    ]
    if trainable_layers == 5:
        layers_to_train.append("bn1")
    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    if extra_blocks is None:
        extra_blocks = LastLevelMaxPool()

    if returned_layers is None:
        returned_layers = [1, 2, 3, 4]
    if min(returned_layers) <= 0 or max(returned_layers) >= 5:
        raise ValueError(
            f"Each returned layer should be in the range [1,4]. Got {returned_layers}"
        )
    return_layers = {f"layer{k}": str(v) for v, k in enumerate(returned_layers)}

    in_channels_list = []
    in_channels_list.append(backbone.layer1[-1].conv3.out_channels)
    in_channels_list.append(backbone.layer2[-1].conv3.out_channels)
    in_channels_list.append(backbone.layer3[-1].conv3.out_channels)
    in_channels_list.append(backbone.layer4[-1].conv3.out_channels)

    out_channels = 256
    return BackboneWithFPN(
        backbone,
        return_layers,
        in_channels_list,
        out_channels,
        extra_blocks=extra_blocks,
        norm_layer=norm_layer,
    )
