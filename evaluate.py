import os
import argparse
import torch

import utils.common as utils
from utils.train import validate
from data import cifar10
from data import imagenet
from models.cifar10 import *
from models.imagenet import *


def parse_args():
    parser = argparse.ArgumentParser("Model evaluation")

    parser.add_argument(
        "--data_dir", type=str, default="~/data", help="path to dataset"
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="vgg_16_bn",
        choices=("vgg_16_bn", "resnet_56", "resnet_110", "densenet_40", "resnet_50"),
        help="architecture",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="vgg_16_bn.pt",
        help="checkpoint path",
    )
    parser.add_argument("--batch_size", type=int, default=256, help="batch size")
    parser.add_argument("--gpu", type=str, default="0", help="Select gpu to use")
    parser.add_argument(
        "-r",
        "--rank",
        dest="rank",
        type=int,
        default=0,
        help="use pre-specified rank for all layers",
    )
    parser.add_argument(
        "-cpr",
        "--compress_rate",
        type=str,
        default="[0.]*100",
        help="list of compress rate of each layer",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    logger = utils.get_logger("log.txt")
    logger.info("args = %s", args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # load data
    print("Loading data:")
    if args.arch == "resnet_50":  # imagenet
        data_tmp = imagenet.Data(args)
        val_loader = data_tmp.test_loader
    else:  # default CIFAR-10
        _, val_loader = cifar10.load_data(args.data_dir, args.batch_size)

    # load checkpoint
    logger.info("Loading checkpoint")
    compress_rate = utils.get_cpr(args.compress_rate)
    model = eval(args.arch)(compress_rate=compress_rate, rank=args.rank).cuda()
    ckpt = torch.load(args.ckpt, map_location="cuda:0")
    model.load_state_dict(ckpt["state_dict"])

    # evaluate
    logger.info("Evaluating model:")
    criterion = torch.nn.CrossEntropyLoss().cuda()
    validate(val_loader, model, criterion, logger)


if __name__ == "__main__":
    main()
