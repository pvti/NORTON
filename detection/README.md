# Downstream tasks training scripts

This folder contains reference training scripts for Faster/Mask/Keypoint-RCNN-ResNet50-FPN for object detection, segmentation and keypoint detection.

To execute the example commands below you must install the following:

```
cython
pycocotools
matplotlib
```

You must modify the following flags:

`--data-path=/path/to/coco/dataset`

`--nproc_per_node=<number_of_gpus_available>`

Weights backbone is adopted from the classification task, therefore it should be the compressed resnet50 with coressponding decomposition rank and pruning ratio.

As [recommended](https://github.com/pytorch/vision/blob/87d54c4e583207e7b003d6b59f1e7f49167f68f1/references/detection/train.py#L85) by `torchvision`, default learning rate and batch size values go along with 8xV100. Please modify them to match with your numbers of gpus, *e.g.,* `--nproc_per_node=1 --lr 0.02 -b 2`.

### Faster R-CNN ResNet-50 FPN
```
torchrun --nproc_per_node=8 train.py --dataset coco --model fasterrcnn_CPresnet50_fpn --epochs 26 --lr-steps 16 22 --aspect-ratio-group-factor 3 --weights-backbone resnet_50_pabs_.0.+.0.1.2+.0.4.5+.0.75.12_1.pt -dr 1 -cpr [0.]+[0.1]*2+[0.4]*5+[0.75]*12
```

### Mask R-CNN
```
torchrun --nproc_per_node=8 train.py --dataset coco --model maskrcnn_CPresnet50_fpn --epochs 26 --lr-steps 16 22 --aspect-ratio-group-factor 3 --weights-backbone resnet_50_pabs_.0.+.0.1.2+.0.4.5+.0.75.12_1.pt -dr 1 -cpr [0.]+[0.1]*2+[0.4]*5+[0.75]*12
```


### Keypoint R-CNN
```
torchrun --nproc_per_node=8 train.py --dataset coco_kp --model keypointrcnn_CPresnet50_fpn --epochs 46 --lr-steps 36 43 --aspect-ratio-group-factor 3 --weights-backbone resnet_50_pabs_.0.+.0.1.2+.0.4.5+.0.75.12_1.pt -dr 1 -cpr [0.]+[0.1]*2+[0.4]*5+[0.75]*12
```


# Visualizing model inference
Compressed models can be deployed as [torchvision's guide](https://pytorch.org/vision/stable/models.html#object-detection-instance-segmentation-and-person-keypoint-detection).

For your convenience, we have also prepared an example script, [visualize.py](./visualize.py), that emphasizes the enhanced FPS achieved by the compressed model. Below is the example usage:

```
python visualize.py --input barbie.mp4 --fps 20 --custom --weight FasterRCNNResNet50FPN_resnet_50_pabs_.0.+.0.1.2+.0.4.5+.0.75.12_1.pt -dr 1 -cpr [0.]+[0.1]*2+[0.4]*5+[0.75]*12
```

By using this script, you can effortlessly visualize and compare the inference speed of both the baseline and pruned models. This provides a clear demonstration of the substantial throughput acceleration achieved by NORTON's compression techniques.
