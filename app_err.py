import os
import datetime
import argparse
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
import torch.utils.data

import utils.common as utils
from utils.train import train, validate
from data import cifar10
from models.cifar10 import *


def parse_args():
    parser = argparse.ArgumentParser('Cifar-10 compute approximation error')

    parser.add_argument('--data_dir', type=str, default='../data',
                        help='path to dataset')
    parser.add_argument('--arch', type=str, default='vgg_16_bn',
                        choices=('vgg_16_bn', 'resnet_56', 'densenet_40'), help='architecture')
    parser.add_argument('--ori_ckpt', type=str, default='checkpoint/cifar10/vgg_16_bn.pt',
                        help='checkpoint path')
    parser.add_argument('--ckpt', type=str, default='checkpoint/cifar10/decomposed/vgg/vgg_16_bn_[0.]*100_1.pt',
                        help='checkpoint path')
    parser.add_argument('--job_dir', type=str, default='app_err',
                        help='path for saving models')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size')
    parser.add_argument('--gpu', type=str, default='0',
                        help='Select gpu to use')
    parser.add_argument('-r', '--rank', dest='rank', type=int, default=1,
                        help='use pre-specified rank for all layers')

    return parser.parse_args()


args = parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

args.job_dir = os.path.join(args.job_dir, args.arch, str(args.rank))
if not os.path.isdir(args.job_dir):
    os.makedirs(args.job_dir)

utils.record_config(args)
now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
logger = utils.get_logger(os.path.join(args.job_dir, now+'.txt'))


def main():
    logger.info('args = %s', args)

    # setup
    train_loader, val_loader = cifar10.load_data(
        args.data_dir, args.batch_size)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    cudnn.benchmark = True
    cudnn.enabled = True

    # load model
    logger.info('Loading baseline model')
    ori_model = eval(args.arch)().cuda()
    ori_ckpt = torch.load(args.ori_ckpt, map_location='cuda:0')
    ori_model.load_state_dict(ori_ckpt['state_dict'])

    logger.info('Loading decomposed model')
    model = eval(args.arch)(rank=args.rank).cuda()
    ckpt = torch.load(args.ckpt, map_location='cuda:0')
    model.load_state_dict(ckpt['state_dict'])

    ori_model.eval()
    model.eval()
    sum = 0
    with torch.no_grad():
        for i, (images, target) in tqdm(enumerate(val_loader)):
            images = images.cuda()
            # compute output
            ori_out = ori_model(images)
            cur_out = model(images)
            err = mse(ori_out, cur_out)
            # logger.info(err)
            sum += err

    logger.info(sum/(i+1))


def mse(x, y):

    return (torch.norm(x-y, p=2).item()/torch.norm(x, p=2).item())**2


if __name__ == '__main__':
    main()
