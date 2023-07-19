import os
import datetime
import argparse
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.utils.data

import utils.common as utils
from models.cifar10 import *

from decomposition.decomposition import cpdblock_weights_to_factors
import tensorly as tl

tl.set_backend("pytorch")


def parse_args():
    parser = argparse.ArgumentParser("Cifar-10 compute approximation error")

    parser.add_argument(
        "--data_dir", type=str, default="../data", help="path to dataset"
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="vgg_16_bn",
        choices=("vgg_16_bn", "resnet_56", "densenet_40"),
        help="architecture",
    )
    parser.add_argument(
        "--ori_ckpt",
        type=str,
        default="checkpoint/cifar10/vgg_16_bn.pt",
        help="checkpoint path",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="checkpoint/cifar10/decomposed/vgg/vgg_16_bn_[0.]*100_1.pt",
        help="checkpoint path",
    )
    parser.add_argument(
        "--job_dir", type=str, default="NMSE", help="path for saving models"
    )
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--gpu", type=str, default="0", help="Select gpu to use")
    parser.add_argument(
        "-r",
        "--rank",
        dest="rank",
        type=int,
        default=1,
        help="use pre-specified rank for all layers",
    )

    return parser.parse_args()


args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

args.job_dir = os.path.join(args.job_dir, args.arch, str(args.rank))
if not os.path.isdir(args.job_dir):
    os.makedirs(args.job_dir)

utils.record_config(args)
now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
logger = utils.get_logger(os.path.join(args.job_dir, now + ".txt"))


def main():
    logger.info("args = %s", args)

    # load model
    logger.info("Loading baseline model")
    ori_model = eval(args.arch)().cuda()
    ori_ckpt = torch.load(args.ori_ckpt, map_location="cuda:0")
    ori_model.load_state_dict(ori_ckpt["state_dict"])

    logger.info("Loading decomposed model")
    model = eval(args.arch)(rank=args.rank).cuda()
    ckpt = torch.load(args.ckpt, map_location="cuda:0")
    model.load_state_dict(ckpt["state_dict"])
    # print(model)

    ori_model.eval()
    model.eval()
    sum = 0
    # print(ori_model._modules['features']._modules['conv0'])

    with torch.no_grad():
        num_layers = 0
        for name, module in ori_model.features.named_modules():
            if isinstance(module, nn.Conv2d):
                num_layers += 1
                # print(name, module)
                # print(model.features._modules[name])
                logger.info(f"processing {name}")

                ori_weight = module.weight
                dcp_module = model.features._modules[name]
                pointwise = dcp_module.feature.pointwise
                vertical = dcp_module.feature.vertical
                horizontal = dcp_module.feature.horizontal
                # print(horizontal)
                head_factor, body_factor, tail_factor = cpdblock_weights_to_factors(
                    pointwise.weight, vertical.weight, horizontal.weight, args.rank
                )
                # print(head_factor.shape, body_factor.shape, tail_factor.shape)
                num_filters = ori_weight.size(0)
                layer_err = 0
                for i in range(num_filters):
                    head = head_factor[:, :, i]
                    body = body_factor[:, :, i]
                    tail = tail_factor[:, :, i]
                    factors = [head, body, tail]
                    reconstruction = tl.cp_to_tensor(cp_tensor=(None, factors))
                    # print(reconstruction.shape)
                    ori_filter = ori_weight[i]
                    err = mse(ori_filter, reconstruction)
                    # logger.info(err)
                    layer_err += err

                layer_err = layer_err / num_filters
                logger.info(f"layer_err = {layer_err}")

                sum += layer_err

    logger.info(f"NMSE = {sum/num_layers}")


def mse(x, y):
    return (torch.norm(x - y, p=2).item() / torch.norm(x, p=2).item()) ** 2


if __name__ == "__main__":
    main()
