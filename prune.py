import os
import datetime
import argparse
import copy
import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
import torch.utils.data

import utils.common as utils
from data import cifar10
from models.cifar10.vgg import vgg_16_bn
from models.cifar10.resnet import resnet_56
from decomposition.CPDLayers import CPDLayer
from saliency import get_saliency


def parse_args():
    parser = argparse.ArgumentParser('Cifar-10 pruning')

    parser.add_argument('--data_dir', type=str, default='../data',
                        help='path to dataset')
    parser.add_argument('--arch', type=str, default='vgg_16_bn',
                        choices=('vgg_16_bn', 'resnet_56'), help='architecture')
    parser.add_argument('--ckpt', type=str,
                        default='result/vgg_16_bn/6/[0.]*100/vgg_16_bn_[0.]*100_6.pt',
                        help='checkpoint path')
    parser.add_argument('--job_dir', type=str, default='result',
                        help='path for saving models')
    parser.add_argument('--batch_size', type=int,
                        default=256, help='batch size')
    parser.add_argument('--epochs', type=int, default=400,
                        help='num of fine-tuning epochs')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='init learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-3,
                        help='weight decay')
    parser.add_argument('--gpu', type=str, default='0',
                        help='Select gpu to use')
    parser.add_argument('-r', '--rank', dest='rank', type=int, default=6,
                        help='use pre-specified rank for all layers')
    parser.add_argument('-cpr', '--compress_rate', type=str, default='[0.]*100',
                        help='list of compress rate of each layer')
    parser.add_argument('--criterion', type=str, default='csa',
                        choices=('csa', 'vbd'), help='criterion for similarity measure')
    parser.add_argument('--name', type=str, default='',
                        help='wandb project name')

    return parser.parse_args()


args = parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
if len(args.gpu) > 1:
    name_base = 'module.'
else:
    name_base = ''

print_freq = (256*50)//args.batch_size

args.job_dir = os.path.join(args.job_dir, args.arch,
                            str(args.rank), args.criterion, args.compress_rate)
if not os.path.isdir(args.job_dir):
    os.makedirs(args.job_dir)

utils.record_config(args)
now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
logger = utils.get_logger(os.path.join(args.job_dir, now+'.txt'))


def prune_vgg(model, ori_state_dict):
    state_dict = model.state_dict()
    last_select_index = None

    cnt = 0
    for name, module in model.named_modules():
        name = name.replace('module.', '')

        if isinstance(module, CPDLayer):
            cnt += 1
            conv_head_weight_name = name + '.head.weight'
            conv_body_weight_name = name + '.body.weight'
            conv_tail_weight_name = name + '.tail.weight'
            ori_head_weight = ori_state_dict[conv_head_weight_name]
            ori_body_weight = ori_state_dict[conv_body_weight_name]
            ori_tail_weight = ori_state_dict[conv_tail_weight_name]
            cur_head_weight = state_dict[conv_head_weight_name]
            # CPDHead has weight tensor of shape (in_channels, out_channels, rank)
            orifilter_num = ori_head_weight.size(1)
            currentfilter_num = cur_head_weight.size(1)

            if orifilter_num != currentfilter_num:
                cov_id = cnt
                logger.info(f'computing saliency for cov_id = {cov_id} ')
                tmp_head_weight = ori_head_weight.detach().clone()
                tmp_head_weight = tmp_head_weight.transpose(1, 0)
                saliency = get_saliency(
                    tmp_head_weight, ori_body_weight, ori_tail_weight, args.criterion)
                select_index = np.argsort(
                    saliency)[orifilter_num-currentfilter_num:]
                select_index.sort()

                if last_select_index is not None:
                    for index_i, i in enumerate(select_index):
                        # CPDBody and CPDTail has weight tensor of shape (out_channels, rank, kernel_size). They don't have in_channels!
                        state_dict[name_base +
                                   conv_body_weight_name][index_i] = ori_state_dict[conv_body_weight_name][i]
                        state_dict[name_base +
                                   conv_tail_weight_name][index_i] = ori_state_dict[conv_tail_weight_name][i]
                        for index_j, j in enumerate(last_select_index):
                            # CPDHead has weight tensor of shape (in_channels, out_channels, rank)
                            state_dict[name_base+conv_head_weight_name][index_j][index_i] = ori_state_dict[conv_head_weight_name][j][i]
                else:
                    for index_i, i in enumerate(select_index):
                        # CPDHead has weight tensor of shape (in_channels, out_channels, rank)
                        state_dict[name_base+conv_head_weight_name][:,
                                                                    index_i] = ori_state_dict[conv_head_weight_name][:, i]
                        # CPDBody and CPDTail has weight tensor of shape (out_channels, rank, kernel_size)
                        state_dict[name_base +
                                   conv_body_weight_name][index_i] = ori_state_dict[conv_body_weight_name][i]
                        state_dict[name_base +
                                   conv_tail_weight_name][index_i] = ori_state_dict[conv_tail_weight_name][i]

                last_select_index = select_index

            elif last_select_index is not None:
                print(conv_head_weight_name)
                state_dict[name_base +
                           conv_body_weight_name] = ori_state_dict[conv_body_weight_name]
                state_dict[name_base +
                           conv_tail_weight_name] = ori_state_dict[conv_tail_weight_name]
                for i in range(orifilter_num):
                    for index_j, j in enumerate(last_select_index):
                        state_dict[name_base +
                                   conv_head_weight_name][index_j][i] = ori_state_dict[conv_head_weight_name][j][i]

            else:
                state_dict[name_base +
                           conv_head_weight_name] = ori_state_dict[conv_head_weight_name]
                state_dict[name_base +
                           conv_body_weight_name] = ori_state_dict[conv_body_weight_name]
                state_dict[name_base +
                           conv_tail_weight_name] = ori_state_dict[conv_tail_weight_name]
                last_select_index = None

    model.load_state_dict(state_dict)

    return model


def prune_resnet(model, ori_state_dict, num_layers=56):
    cfg = {
        56: [9, 9, 9],
        110: [18, 18, 18],
    }

    state_dict = model.state_dict()

    current_cfg = cfg[num_layers]
    last_select_index = None

    all_conv_weight = []

    cnt = 1
    for layer, num in enumerate(current_cfg):
        layer_name = 'layer' + str(layer + 1) + '.'
        for k in range(num):
            for l in range(2):
                cnt += 1
                cov_id = cnt

                conv_name = layer_name + str(k) + '.conv' + str(l + 1)
                conv_weight_name = conv_name + '.weight'
                conv_head_weight_name = conv_name + '.head.weight'
                conv_body_weight_name = conv_name + '.body.weight'
                conv_tail_weight_name = conv_name + '.tail.weight'
                all_conv_weight.append(conv_weight_name)
                ori_head_weight = ori_state_dict[conv_head_weight_name]
                ori_body_weight = ori_state_dict[conv_body_weight_name]
                ori_tail_weight = ori_state_dict[conv_tail_weight_name]
                cur_head_weight = state_dict[conv_head_weight_name]
                # CPDHead has weight tensor of shape (in_channels, out_channels, rank)
                orifilter_num = ori_head_weight.size(1)
                currentfilter_num = cur_head_weight.size(1)

                if orifilter_num != currentfilter_num:
                    logger.info(f'computing saliency for cov_id = {cov_id}')
                    tmp_head_weight = ori_head_weight.detach().clone()
                    tmp_head_weight = tmp_head_weight.transpose(1, 0)
                    saliency = get_saliency(
                        tmp_head_weight, ori_body_weight, ori_tail_weight, args.criterion)
                    select_index = np.argsort(
                        saliency)[orifilter_num - currentfilter_num:]
                    select_index.sort()

                    if last_select_index is not None:
                        # current layer, out channel
                        for index_i, i in enumerate(select_index):
                            # CPDBody and CPDTail has weight tensor of shape (out_channels, rank, kernel_size). They don't have in_channels!
                            state_dict[name_base +
                                       conv_body_weight_name][index_i] = ori_state_dict[conv_body_weight_name][i]
                            state_dict[name_base +
                                       conv_tail_weight_name][index_i] = ori_state_dict[conv_tail_weight_name][i]
                            # last layer, in channel
                            for index_j, j in enumerate(last_select_index):
                                # CPDHead has weight tensor of shape (in_channels, out_channels, rank)
                                state_dict[name_base+conv_head_weight_name][index_j][index_i] = ori_state_dict[conv_head_weight_name][j][i]
                    else:
                        for index_i, i in enumerate(select_index):
                            # CPDHead has weight tensor of shape (in_channels, out_channels, rank)
                            state_dict[name_base+conv_head_weight_name][:,
                                                                        index_i] = ori_state_dict[conv_head_weight_name][:, i]
                            # CPDBody and CPDTail has weight tensor of shape (out_channels, rank, kernel_size)
                            state_dict[name_base +
                                       conv_body_weight_name][index_i] = ori_state_dict[conv_body_weight_name][i]
                            state_dict[name_base +
                                       conv_tail_weight_name][index_i] = ori_state_dict[conv_tail_weight_name][i]

                    last_select_index = select_index

                # second conv layers of layer3
                elif last_select_index is not None:
                    print(conv_head_weight_name)
                    state_dict[name_base +
                               conv_body_weight_name] = ori_state_dict[conv_body_weight_name]
                    state_dict[name_base +
                               conv_tail_weight_name] = ori_state_dict[conv_tail_weight_name]
                    for i in range(orifilter_num):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[name_base +
                                       conv_head_weight_name][index_j][i] = ori_state_dict[conv_head_weight_name][j][i]
                    last_select_index = None

                else:
                    state_dict[name_base +
                               conv_head_weight_name] = ori_state_dict[conv_head_weight_name]
                    state_dict[name_base +
                               conv_body_weight_name] = ori_state_dict[conv_body_weight_name]
                    state_dict[name_base +
                               conv_tail_weight_name] = ori_state_dict[conv_tail_weight_name]
                    last_select_index = None

    # fetch all remaining layers (Conv2d/CPDLayer/Linear) that is nof affect from pruning, e.g. ratio = 0. Ignore BN layers
    for name, module in model.named_modules():
        name = name.replace('module.', '')

        if isinstance(module, nn.Conv2d):
            conv_name = name + '.weight'
            if 'shortcut' in name:
                continue
            if conv_name not in all_conv_weight:
                state_dict[name_base+conv_name] = ori_state_dict[conv_name]

        elif isinstance(module, nn.Linear):
            state_dict[name_base+name +
                       '.weight'] = ori_state_dict[name + '.weight']
            state_dict[name_base+name +
                       '.bias'] = ori_state_dict[name + '.bias']

    model.load_state_dict(state_dict)

    return model


def main():
    logger.info('args = %s', args)
    # init wandb
    name = f'{args.criterion}_{args.compress_rate}_{args.rank}'
    wandb.init(name=name,
               project=f'NORTON_Prune_{args.name}_{args.arch}',
               config=vars(args))

    # setup
    train_loader, val_loader = cifar10.load_data(
        args.data_dir, args.batch_size)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    cudnn.benchmark = True
    cudnn.enabled = True

    compress_rate = utils.get_cpr(args.compress_rate)

    # load decomposed model
    logger.info('Loading decomposed model')
    ori_model = eval(args.arch)(compress_rate=[0.]*100, rank=args.rank).cuda()
    ckpt = torch.load(args.ckpt, map_location='cuda:0')
    ori_model.load_state_dict(ckpt['state_dict'])
    ori_state_dict = ori_model.state_dict()

    # prune
    logger.info('Pruning model:')
    model = eval(args.arch)(compress_rate=compress_rate, rank=args.rank).cuda()
    logger.info(model)
    if args.arch == 'vgg_16_bn':
        prune_vgg(model, ori_state_dict)
    elif args.arch == 'resnet_56':
        prune_resnet(model, ori_state_dict, 56)

    # finetune
    logger.info('Finetuning model:')
    model = finetune(model, train_loader, val_loader, args.epochs)

    # save model
    path = os.path.join(args.job_dir, f'{args.arch}_{name}.pt')
    torch.save({'state_dict': model.state_dict(),
                'rank': args.rank,
                'compress_rate': args.compress_rate},
               path)


def train(epoch, train_loader, model, criterion, optimizer, scheduler):
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')

    model.train()

    cur_lr = optimizer.param_groups[0]['lr']
    logger.info('learning_rate: ' + str(cur_lr))

    num_iter = len(train_loader)
    for i, (images, target) in enumerate(train_loader):
        images = images.cuda()
        target = target.cuda()

        # compute output
        logits = model(images)
        loss = criterion(logits, target)

        # measure accuracy and record loss
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = images.size(0)
        losses.update(loss.item(), n)  # accumulated loss
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % print_freq == 0:
            logger.info(
                'Epoch[{0}]({1}/{2}): '
                'Loss {loss.avg:.4f} '
                'Prec@1(1,5) {top1.avg:.2f}, {top5.avg:.2f} '
                'Lr {cur_lr:.4f}'.format(
                    epoch, i, num_iter, loss=losses,
                    top1=top1, top5=top5, cur_lr=cur_lr))
    scheduler.step()

    return losses.avg, top1.avg, top5.avg


def validate(val_loader, model, criterion):
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')

    # switch to evaluation mode
    model.eval()
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()

            # compute output
            logits = model(images)
            loss = criterion(logits, target)

            # measure accuracy and record loss
            pred1, pred5 = utils.accuracy(logits, target, topk=(1, 5))
            n = images.size(0)
            losses.update(loss.item(), n)
            top1.update(pred1[0], n)
            top5.update(pred5[0], n)

        logger.info(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                    .format(top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg


def finetune(model, train_loader, val_loader, epochs):
    criterion = nn.CrossEntropyLoss()
    # use a small learning rate
    optimizer = torch.optim.SGD(model.parameters(
    ), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs)

    _, best_top1_acc, _ = validate(val_loader, model, criterion)
    best_model_state = copy.deepcopy(model.state_dict())
    epoch = 0
    while epoch < epochs:
        train(epoch, train_loader, model, criterion, optimizer, scheduler)
        _, valid_top1_acc, _ = validate(val_loader, model, criterion)

        if valid_top1_acc > best_top1_acc:
            best_top1_acc = valid_top1_acc
            best_model_state = copy.deepcopy(model.state_dict())

        cur_lr = optimizer.param_groups[0]['lr']
        wandb.log({'best_acc': max(valid_top1_acc, best_top1_acc),
                  'top1': valid_top1_acc, 'lr': cur_lr})

        epoch += 1
        logger.info('=>Best accuracy {:.3f}'.format(best_top1_acc))

    model.load_state_dict(best_model_state)

    return model


if __name__ == '__main__':
    main()
