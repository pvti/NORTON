import os
import datetime
import argparse
import copy
import wandb
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
    parser = argparse.ArgumentParser('Cifar-10 sweep training from scratch')

    parser.add_argument('--data_dir', type=str, default='../data',
                        help='path to dataset')
    parser.add_argument('--arch', type=str, default='vgg_16_bn',
                        choices=('vgg_16_bn', 'resnet_56'), help='architecture')
    parser.add_argument('--ckpt', type=str, default='checkpoint/cifar10/vgg_16_bn.pt',
                        help='checkpoint path')
    parser.add_argument('--job_dir', type=str, default='result',
                        help='path for saving models')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch size')
    parser.add_argument('--epochs', type=int, default=400,
                        help='num of fine-tuning epochs')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='init learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='weight decay')
    parser.add_argument('--gpu', type=str, default='0',
                        help='Select gpu to use')
    parser.add_argument('-r', '--rank', dest='rank', type=int, default=1,
                        help='use pre-specified rank for all layers')
    parser.add_argument('-cpr', '--compress_rate', type=str, default='[0.]*100',
                        help='list of compress rate of each layer')
    parser.add_argument('--name', type=str, default='',
                        help='wandb project name')

    return parser.parse_args()


args = parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

args.job_dir = os.path.join(args.job_dir, args.arch,
                            str(args.rank), args.compress_rate)
if not os.path.isdir(args.job_dir):
    os.makedirs(args.job_dir)

utils.record_config(args)
now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
logger = utils.get_logger(os.path.join(args.job_dir, now+'.txt'))

sweep_configuration = {
        'method': 'grid',
        'name': 'sweep',
        'metric': {
            'goal': 'maximize',
            'name': 'top1'
        },
        'parameters': {
            'batch_size': {'values': [128, 256, 512]},
            'lr': {'values': [0.1, 0.05, 0.01, 0.005, 0.001]},
            'weight_decay': {'values': [5e-3, 5e-4]}
        }
    }

sweep_id = wandb.sweep(sweep=sweep_configuration, project="Sweep training from scratch")


def main():
    run = wandb.init()

    args.batch_size = wandb.config.batch_size
    args.lr = wandb.config.lr
    args.weight_decay = wandb.config.weight_decay

    logger.info('args = %s', args)

    # setup
    train_loader, val_loader = cifar10.load_data(
        args.data_dir, args.batch_size)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    cudnn.benchmark = True
    cudnn.enabled = True

    # load model
    logger.info('Loading baseline or pruned model')
    compress_rate = utils.get_cpr(args.compress_rate)
    model = eval(args.arch)(compress_rate=compress_rate, rank=args.rank).cuda()
    logger.info(model)

    # finetune
    logger.info('Finetuning model:')
    model, best_top1_acc = finetune(model, train_loader, val_loader, args.epochs, criterion)

    # save model
    name = f'{args.compress_rate}_{args.rank}_{args.batch_size}_{args.lr}_{args.weight_decay}'
    path = os.path.join(args.job_dir, f'{args.arch}_{name}_{best_top1_acc}.pt')
    torch.save({'state_dict': model.state_dict(),
                'rank': args.rank},
               path)


def finetune(model, train_loader, val_loader, epochs, criterion):
    # use a small learning rate
    optimizer = torch.optim.SGD(model.parameters(
    ), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs)
    # lr_decay_step = list(map(int, args.lr_decay_step.split(',')))
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #     optimizer, milestones=lr_decay_step, gamma=0.1)

    _, best_top1_acc, _ = validate(val_loader, model, criterion, logger)
    best_model_state = copy.deepcopy(model.state_dict())
    epoch = 0
    while epoch < epochs:
        train(epoch, train_loader, model, criterion, optimizer, scheduler, logger)
        _, valid_top1_acc, _ = validate(val_loader, model, criterion, logger)

        if valid_top1_acc > best_top1_acc:
            best_top1_acc = valid_top1_acc
            best_model_state = copy.deepcopy(model.state_dict())

        cur_lr = optimizer.param_groups[0]['lr']
        wandb.log({'best_acc': max(valid_top1_acc, best_top1_acc),
                  'top1': valid_top1_acc, 'lr': cur_lr})

        epoch += 1
        logger.info('=>Best accuracy {:.3f}'.format(best_top1_acc))

    model.load_state_dict(best_model_state)

    return model, best_top1_acc


if __name__ == '__main__':
    wandb.agent(sweep_id, function=main, count=100)
