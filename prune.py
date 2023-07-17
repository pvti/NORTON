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
from decomposition.CPDBlock import CPDBlock
from pruning.prune import prune_factors
from decomposition.decomposition import cpdblock_weights_to_factors, factors_to_cpdblock_weights


def parse_args():
    parser = argparse.ArgumentParser('Cifar-10 pruning')

    parser.add_argument('--data_dir', type=str, default='../data',
                        help='path to dataset')
    parser.add_argument('--arch', type=str, default='vgg_16_bn',
                        choices=('vgg_16_bn', 'resnet_56', 'resnet_110', 'densenet_40'),
                        help='architecture')
    parser.add_argument('--ckpt', type=str,
                        default='result/vgg_16_bn/6/[0.]*100/vgg_16_bn_[0.]*100_6.pt',
                        help='checkpoint path')
    parser.add_argument('--job_dir', type=str, default='result',
                        help='path for saving models')
    parser.add_argument('--batch_size', type=int,
                        default=256, help='batch size')
    parser.add_argument('--epochs', type=int, default=400,
                        help='num of fine-tuning epochs')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='init learning rate')
    parser.add_argument("--lr-warmup-epochs", default=5, type=int,
                        help="the number of epochs to warmup (default: 5)")
    parser.add_argument("--lr-warmup-decay", default=0.01, type=float,
                        help="the decay for lr")
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='weight decay')
    parser.add_argument('--gpu', type=str, default='0',
                        help='Select gpu to use')
    parser.add_argument('-r', '--rank', dest='rank', type=int, default=6,
                        help='use pre-specified rank for all layers')
    parser.add_argument('-cpr', '--compress_rate', type=str, default='[0.]*100',
                        help='list of compress rate of each layer')
    parser.add_argument('--criterion', type=str, default='pabs',
                        choices=('pabs', 'csa', 'vbd'), help='criterion for similarity measure')
    parser.add_argument('--name', type=str, default='',
                        help='wandb project name')

    return parser.parse_args()


args = parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
if len(args.gpu) > 1:
    name_base = 'module.'
else:
    name_base = ''

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

    for name, module in model.named_modules():
        name = name.replace('module.', '')

        if isinstance(module, CPDBlock):
            pointwise_weight_name = name + '.feature.pointwise.weight'
            vertical_weight_name = name + '.feature.vertical.weight'
            horizontal_weight_name = name + '.feature.horizontal.weight'
            bias_name = name + '.bias'
            ori_pointwise_weight = ori_state_dict[pointwise_weight_name]
            ori_vertical_weight = ori_state_dict[vertical_weight_name]
            ori_horizontal_weight = ori_state_dict[horizontal_weight_name]
            cur_pointwise_weight = state_dict[pointwise_weight_name]

            # Pointwise module has weight tensor of shape (rank*out_channels, in_channels)
            ori_num_filter = ori_pointwise_weight.size(0)
            cur_num_filter = cur_pointwise_weight.size(0)

            # number of filters in conv2d form, be careful
            ori_out_channels = int(ori_num_filter / args.rank)
            cur_out_channels = int(cur_num_filter / args.rank)

            # out_channels changes
            if ori_out_channels != cur_out_channels:
                logger.info(f'computing saliency for {name} ')
                ori_head_factor, ori_body_factor, ori_tail_factor = cpdblock_weights_to_factors(
                    ori_pointwise_weight, ori_vertical_weight, ori_horizontal_weight, args.rank)

                # update original head factor if in_channels changed.
                updated_head_factor = ori_head_factor
                if last_select_index is not None:
                    cur_pointwise_in_channels = cur_pointwise_weight.size(1)
                    updated_head_factor = torch.empty(
                        (cur_pointwise_in_channels, ori_head_factor.size(1), ori_head_factor.size(2)))
                    for index_i, i in enumerate(last_select_index):
                        updated_head_factor[index_i] = ori_head_factor[i]

                head_factor, body_factor, tail_factor, select_index = prune_factors(
                    updated_head_factor, ori_body_factor, ori_tail_factor, cur_out_channels, args.criterion)
                pointwise_weight, vertical_weight, horizontal_weight = factors_to_cpdblock_weights(
                    head_factor, body_factor, tail_factor)
                state_dict[name_base +
                           pointwise_weight_name] = pointwise_weight
                state_dict[name_base + vertical_weight_name] = vertical_weight
                state_dict[name_base +
                           horizontal_weight_name] = horizontal_weight

                for index_i, i in enumerate(select_index):
                    state_dict[name_base +
                               bias_name][index_i] = ori_state_dict[bias_name][i]

                last_select_index = select_index

            # out_channels is identical but in_channels changed
            elif last_select_index is not None:
                logger.info(f'treat {name} which is not pruned')
                state_dict[name_base +
                           vertical_weight_name] = ori_state_dict[vertical_weight_name]
                state_dict[name_base +
                           horizontal_weight_name] = ori_state_dict[horizontal_weight_name]
                for i in range(ori_num_filter):
                    for index_j, j in enumerate(last_select_index):
                        state_dict[name_base +
                                   pointwise_weight_name][i][index_j] = ori_state_dict[pointwise_weight_name][i][j]
                last_select_index = None

            # none changes
            else:
                state_dict[name_base +
                           pointwise_weight_name] = ori_state_dict[pointwise_weight_name]
                state_dict[name_base +
                           vertical_weight_name] = ori_state_dict[vertical_weight_name]
                state_dict[name_base +
                           horizontal_weight_name] = ori_state_dict[horizontal_weight_name]
                last_select_index = None

    # treat remaining layers (Linear)
    for name, module in model.named_modules():
        name = name.replace('module.', '')
        if isinstance(module, nn.Linear):
            logger.info(f'treat {name} which is not pruned')

            # linear1
            if last_select_index is not None:
                for index_i, i in enumerate(last_select_index):
                    state_dict[name_base+name + '.weight'][:,
                                                           index_i] = ori_state_dict[name + '.weight'][:, i]
                last_select_index = None

            # linear2
            else:
                state_dict[name_base+name +
                           '.weight'] = ori_state_dict[name + '.weight']
            state_dict[name_base+name +
                       '.bias'] = ori_state_dict[name + '.bias']

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

    # list to save processed layer which has compress_rate > 0 or in_channels being affected
    processed_conv = []

    for layer, num in enumerate(current_cfg):
        layer_name = 'layer' + str(layer + 1) + '.'
        for k in range(num):
            for l in range(2):
                conv_name = layer_name + str(k) + '.conv' + str(l + 1)
                pointwise_weight_name = conv_name + '.feature.pointwise.weight'
                vertical_weight_name = conv_name + '.feature.vertical.weight'
                horizontal_weight_name = conv_name + '.feature.horizontal.weight'
                processed_conv.append(conv_name)
                ori_pointwise_weight = ori_state_dict[pointwise_weight_name]
                ori_vertical_weight = ori_state_dict[vertical_weight_name]
                ori_horizontal_weight = ori_state_dict[horizontal_weight_name]
                cur_pointwise_weight = state_dict[pointwise_weight_name]

                # Pointwise module has weight tensor of shape (rank*out_channels, in_channels)
                ori_num_filter = ori_pointwise_weight.size(0)
                cur_num_filter = cur_pointwise_weight.size(0)

                # number of filters in conv2d form, be careful
                ori_out_channels = int(ori_num_filter / args.rank)
                cur_out_channels = int(cur_num_filter / args.rank)

                # out_channels changes
                if ori_out_channels != cur_out_channels:
                    logger.info(f'computing saliency for {conv_name}')
                    ori_head_factor, ori_body_factor, ori_tail_factor = cpdblock_weights_to_factors(
                        ori_pointwise_weight, ori_vertical_weight, ori_horizontal_weight, args.rank)

                    # update original head factor if in_channels changed.
                    updated_head_factor = ori_head_factor
                    if last_select_index is not None:
                        cur_pointwise_in_channels = cur_pointwise_weight.size(
                            1)
                        updated_head_factor = torch.empty(
                            (cur_pointwise_in_channels, ori_head_factor.size(1), ori_head_factor.size(2)))
                        for index_i, i in enumerate(last_select_index):
                            updated_head_factor[index_i] = ori_head_factor[i]

                    head_factor, body_factor, tail_factor, select_index = prune_factors(
                        updated_head_factor, ori_body_factor, ori_tail_factor, cur_out_channels, args.criterion)
                    pointwise_weight, vertical_weight, horizontal_weight = factors_to_cpdblock_weights(
                        head_factor, body_factor, tail_factor)
                    state_dict[name_base +
                               pointwise_weight_name] = pointwise_weight
                    state_dict[name_base +
                               vertical_weight_name] = vertical_weight
                    state_dict[name_base +
                               horizontal_weight_name] = horizontal_weight

                    last_select_index = select_index

                # out_channels is identical but in_channels changed
                elif last_select_index is not None:
                    logger.info(f'treat {conv_name} which is not pruned')
                    state_dict[name_base +
                               vertical_weight_name] = ori_state_dict[vertical_weight_name]
                    state_dict[name_base +
                               horizontal_weight_name] = ori_state_dict[horizontal_weight_name]
                    for i in range(ori_num_filter):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[name_base +
                                       pointwise_weight_name][i][index_j] = ori_state_dict[pointwise_weight_name][i][j]
                    last_select_index = None

                # none changes
                else:
                    logger.info(f'treat {conv_name} which is untouched')
                    state_dict[name_base +
                               pointwise_weight_name] = ori_state_dict[pointwise_weight_name]
                    state_dict[name_base +
                               vertical_weight_name] = ori_state_dict[vertical_weight_name]
                    state_dict[name_base +
                               horizontal_weight_name] = ori_state_dict[horizontal_weight_name]
                    last_select_index = None

    # treat remaining layers which are totally untouched/unprocessed
    for name, module in model.named_modules():
        name = name.replace('module.', '')

        if isinstance(module, CPDBlock):
            if 'shortcut' in name:
                continue
            if name not in processed_conv:
                logger.info(f'treat {name} which is untouched')
                pointwise_weight_name = name + '.feature.pointwise.weight'
                vertical_weight_name = name + '.feature.vertical.weight'
                horizontal_weight_name = name + '.feature.horizontal.weight'
                state_dict[name_base +
                           pointwise_weight_name] = ori_state_dict[pointwise_weight_name]
                state_dict[name_base +
                           vertical_weight_name] = ori_state_dict[vertical_weight_name]
                state_dict[name_base +
                           horizontal_weight_name] = ori_state_dict[horizontal_weight_name]

        elif isinstance(module, nn.Linear):
            logger.info(f'treat {name} which is not pruned')
            state_dict[name_base+name +
                       '.weight'] = ori_state_dict[name + '.weight']
            state_dict[name_base+name +
                       '.bias'] = ori_state_dict[name + '.bias']

    model.load_state_dict(state_dict)

    return model


def prune_densenet(model, ori_state_dict):
    state_dict = model.state_dict()
    last_select_index = []

    cov_id = 0
    for name, module in model.named_modules():
        name = name.replace('module.', '')

        if isinstance(module, CPDBlock):
            cov_id += 1
            pointwise_weight_name = name + '.feature.pointwise.weight'
            vertical_weight_name = name + '.feature.vertical.weight'
            horizontal_weight_name = name + '.feature.horizontal.weight'
            ori_pointwise_weight = ori_state_dict[pointwise_weight_name]
            ori_vertical_weight = ori_state_dict[vertical_weight_name]
            ori_horizontal_weight = ori_state_dict[horizontal_weight_name]
            cur_pointwise_weight = state_dict[pointwise_weight_name]

            # Pointwise module has weight tensor of shape (rank*out_channels, in_channels)
            ori_num_filter = ori_pointwise_weight.size(0)
            cur_num_filter = cur_pointwise_weight.size(0)

            # number of filters in conv2d form, be careful
            ori_out_channels = int(ori_num_filter / args.rank)
            cur_out_channels = int(cur_num_filter / args.rank)

            # out_channels changes
            if ori_out_channels != cur_out_channels:
                logger.info(f'computing saliency for {name}')
                ori_head_factor, ori_body_factor, ori_tail_factor = cpdblock_weights_to_factors(
                    ori_pointwise_weight, ori_vertical_weight, ori_horizontal_weight, args.rank)

                # update original head factor if in_channels changed.
                updated_head_factor = ori_head_factor
                if last_select_index is not None:
                    cur_pointwise_in_channels = cur_pointwise_weight.size(1)
                    updated_head_factor = torch.empty(
                        (cur_pointwise_in_channels, ori_head_factor.size(1), ori_head_factor.size(2)))
                    for index_i, i in enumerate(last_select_index):
                        updated_head_factor[index_i] = ori_head_factor[i]

                head_factor, body_factor, tail_factor, select_index = prune_factors(
                    updated_head_factor, ori_body_factor, ori_tail_factor, cur_out_channels, args.criterion)
                pointwise_weight, vertical_weight, horizontal_weight = factors_to_cpdblock_weights(
                    head_factor, body_factor, tail_factor)
                state_dict[name_base +
                           pointwise_weight_name] = pointwise_weight
                state_dict[name_base + vertical_weight_name] = vertical_weight
                state_dict[name_base +
                           horizontal_weight_name] = horizontal_weight

            # out_channels is identical but in_channels changed
            elif last_select_index is not None:
                logger.info(f'treat {name} which is not pruned')
                state_dict[name_base +
                           vertical_weight_name] = ori_state_dict[vertical_weight_name]
                state_dict[name_base +
                           horizontal_weight_name] = ori_state_dict[horizontal_weight_name]
                for i in range(ori_num_filter):
                    for index_j, j in enumerate(last_select_index):
                        state_dict[name_base +
                                   pointwise_weight_name][i][index_j] = ori_state_dict[pointwise_weight_name][i][j]

                # first conv layer
                if last_select_index == []:
                    state_dict[name_base +
                               pointwise_weight_name] = ori_state_dict[pointwise_weight_name]

                select_index = list(range(0, ori_out_channels))

            # none changes
            else:
                select_index = list(range(0, ori_out_channels))
                state_dict[name_base +
                           pointwise_weight_name] = ori_state_dict[pointwise_weight_name]
                state_dict[name_base +
                           vertical_weight_name] = ori_state_dict[vertical_weight_name]
                state_dict[name_base +
                           horizontal_weight_name] = ori_state_dict[horizontal_weight_name]

            # transition layer
            if cov_id == 1 or cov_id == 14 or cov_id == 27:
                last_select_index = select_index

            else:
                tmp_select_index = [x+cov_id*12 -
                                    (cov_id-1)//13*12 for x in select_index]
                last_select_index.extend(tmp_select_index)

    # treat remaining layers (Linear)
    for name, module in model.named_modules():
        name = name.replace('module.', '')
        if isinstance(module, nn.Linear):
            logger.info(f'treat {name} which is not pruned')
            for index_i, i in enumerate(last_select_index):
                state_dict[name_base+name + '.weight'][:,
                                                       index_i] = ori_state_dict[name + '.weight'][:, i]
            state_dict[name_base+name +
                       '.bias'] = ori_state_dict[name + '.bias']

    model.load_state_dict(state_dict)


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
    elif args.arch == 'resnet_110':
        prune_resnet(model, ori_state_dict, 110)
    elif args.arch == 'densenet_40':
        prune_densenet(model, ori_state_dict)

    # finetune
    logger.info('Finetuning model:')
    model = finetune(model, train_loader, val_loader, args.epochs, criterion)

    # save model
    path = os.path.join(args.job_dir, f'{args.arch}_{name}.pt')
    torch.save({'state_dict': model.state_dict(),
                'rank': args.rank,
                'compress_rate': args.compress_rate},
               path)


def finetune(model, train_loader, val_loader, epochs, criterion):
    optimizer = torch.optim.SGD(model.parameters(
    ), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs-args.lr_warmup_epochs)
    warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.lr_warmup_epochs])

    _, best_top1_acc, _ = validate(val_loader, model, criterion, logger)
    best_model_state = copy.deepcopy(model.state_dict())
    epoch = 0
    while epoch < epochs:
        train(epoch, train_loader, model, criterion,
              optimizer, scheduler, logger)
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

    return model


if __name__ == '__main__':
    main()
