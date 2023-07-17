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
from data import imagenet
from models.imagenet import *
from decomposition.CPDBlock import CPDBlock
from pruning.prune import prune_factors, prune_conv
from decomposition.decomposition import decompose, cpdblock_weights_to_factors, factors_to_cpdblock_weights


def parse_args():
    parser = argparse.ArgumentParser(
        'Imagenet decomposition, pruning and finetuning')

    parser.add_argument('--data_dir', type=str, default='../data/imagenet/',
                        help='path to dataset')
    parser.add_argument('--arch', type=str, default='resnet_50',
                        choices=('resnet_50'),
                        help='architecture')
    parser.add_argument('--ckpt', type=str,
                        default='checkpoint/imagenet/resnet_50.pt',
                        help='checkpoint path')
    parser.add_argument('--job_dir', type=str, default='result',
                        help='path for saving models')
    parser.add_argument('--batch_size', type=int,
                        default=256, help='batch size')
    parser.add_argument('--epochs', type=int, default=200,
                        help='num of fine-tuning epochs')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='init learning rate')
    parser.add_argument("--lr-warmup-epochs", default=5, type=int,
                        help="the number of epochs to warmup (default: 5)")
    parser.add_argument("--lr-warmup-decay", default=0.01, type=float,
                        help="the decay for lr")
    parser.add_argument('--momentum', type=float, default=0.99,
                        help='momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--label_smooth', type=float, default=0.1,
                        help='label smoothing')
    parser.add_argument('--gpu', type=str, default='0',
                        help='Select gpu to use')
    parser.add_argument('-r', '--rank', dest='rank', type=int, default=6,
                        help='use pre-specified rank for all layers')
    parser.add_argument('-cpr', '--compress_rate', type=str, default='[0.]*100',
                        help='list of compress rate of each layer')
    parser.add_argument('--criterion', type=str, default='pabs',
                        choices=('pabs', 'csa', 'vbd'), help='criterion for similarity measure')
    parser.add_argument('--n_iter_max', type=int, default=300,
                        help='max number of iterations for parafac')
    parser.add_argument('--n_iter_singular_error', type=int, default=3,
                        help='number of iterations for singular maxtrix error handler')
    parser.add_argument('--name', type=str, default='',
                        help='wandb project name')

    return parser.parse_args()


args = parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

args.job_dir = os.path.join(args.job_dir, args.arch,
                            str(args.rank), args.criterion, args.compress_rate)
if not os.path.isdir(args.job_dir):
    os.makedirs(args.job_dir)

utils.record_config(args)
now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
logger = utils.get_logger(os.path.join(args.job_dir, now+'.txt'))


def prune_resnet(model, ori_state_dict):
    state_dict = model.state_dict()

    current_cfg = [3, 4, 6, 3]
    last_select_index = None

    bn_part_name = ['.weight', '.bias', '.running_mean', '.running_var']
    for layer, num in enumerate(current_cfg):
        layer_name = 'layer' + str(layer + 1) + '.'

        for k in range(num):
            iter = 3
            if k == 0:
                iter += 1
            for l in range(iter):
                record_last = True
                if k == 0 and l == 2:
                    conv_name = layer_name + str(k) + '.downsample.0'
                    bn_name = layer_name + str(k) + '.downsample.1'
                    record_last = False
                elif k == 0 and l == 3:
                    conv_name = layer_name + str(k) + '.conv' + str(l)
                    bn_name = layer_name + str(k) + '.bn' + str(l)
                else:
                    conv_name = layer_name + str(k) + '.conv' + str(l + 1)
                    bn_name = layer_name + str(k) + '.bn' + str(l + 1)

                # CPDBlock
                if 'conv2' in conv_name:
                    pointwise_weight_name = conv_name + '.feature.pointwise.weight'
                    vertical_weight_name = conv_name + '.feature.vertical.weight'
                    horizontal_weight_name = conv_name + '.feature.horizontal.weight'
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
                        state_dict[pointwise_weight_name] = pointwise_weight
                        state_dict[vertical_weight_name] = vertical_weight
                        state_dict[horizontal_weight_name] = horizontal_weight

                        if record_last:
                            last_select_index = select_index

                    # out_channels is identical but in_channels changed
                    elif last_select_index is not None:
                        logger.info(f'treat {conv_name} which is not pruned')
                        state_dict[vertical_weight_name] = ori_state_dict[vertical_weight_name]
                        state_dict[horizontal_weight_name] = ori_state_dict[horizontal_weight_name]
                        for i in range(ori_num_filter):
                            for index_j, j in enumerate(last_select_index):
                                state_dict[pointwise_weight_name][i][index_j] = ori_state_dict[pointwise_weight_name][i][j]
                        if record_last:
                            last_select_index = None

                    # none changes
                    else:
                        logger.info(f'treat {conv_name} which is untouched')
                        state_dict[pointwise_weight_name] = ori_state_dict[pointwise_weight_name]
                        state_dict[vertical_weight_name] = ori_state_dict[vertical_weight_name]
                        state_dict[horizontal_weight_name] = ori_state_dict[horizontal_weight_name]
                        if record_last:
                            last_select_index = None

                # Conv2d
                else:
                    conv_weight_name = conv_name + '.weight'
                    oriweight = ori_state_dict[conv_weight_name]
                    curweight = state_dict[conv_weight_name]
                    orifilter_num = oriweight.size(0)
                    currentfilter_num = curweight.size(0)

                    if orifilter_num != currentfilter_num:
                        logger.info(f'computing saliency for {conv_name}')
                        select_index = prune_conv(
                            oriweight, currentfilter_num, args.criterion)

                        if last_select_index is not None:
                            for index_i, i in enumerate(select_index):
                                for index_j, j in enumerate(last_select_index):
                                    state_dict[conv_weight_name][index_i][index_j] = \
                                        ori_state_dict[conv_weight_name][i][j]

                                for bn_part in bn_part_name:
                                    state_dict[bn_name + bn_part][index_i] = \
                                        ori_state_dict[bn_name + bn_part][i]

                        else:
                            for index_i, i in enumerate(select_index):
                                state_dict[conv_weight_name][index_i] = \
                                    ori_state_dict[conv_weight_name][i]

                                for bn_part in bn_part_name:
                                    state_dict[bn_name + bn_part][index_i] = \
                                        ori_state_dict[bn_name + bn_part][i]

                        if record_last:
                            last_select_index = select_index

                    elif last_select_index is not None:
                        for index_i in range(orifilter_num):
                            for index_j, j in enumerate(last_select_index):
                                state_dict[conv_weight_name][index_i][index_j] = \
                                    ori_state_dict[conv_weight_name][index_i][j]

                        for bn_part in bn_part_name:
                            state_dict[bn_name + bn_part] = \
                                ori_state_dict[bn_name + bn_part]

                        if record_last:
                            last_select_index = None

                    else:
                        state_dict[conv_weight_name] = oriweight
                        for bn_part in bn_part_name:
                            state_dict[bn_name + bn_part] = \
                                ori_state_dict[bn_name + bn_part]
                        if record_last:
                            last_select_index = None

                state_dict[bn_name + '.num_batches_tracked'] = ori_state_dict[bn_name +
                                                                              '.num_batches_tracked']

    # treat remaining layers which are totally untouched/unprocessed
    for name, module in model.named_modules():
        name = name.replace('module.', '')
        if isinstance(module, nn.Linear):
            logger.info(f'treat {name} which is untouched')
            state_dict[name + '.weight'] = ori_state_dict[name + '.weight']
            state_dict[name + '.bias'] = ori_state_dict[name + '.bias']

    model.load_state_dict(state_dict)

    return model


def main():
    logger.info('args = %s', args)
    # init wandb
    name = f'{args.criterion}_{args.compress_rate}_{args.rank}'
    wandb.init(name=name,
               project=f'NORTON_Decompose_Prune_{args.name}_{args.arch}',
               config=vars(args))

    # criterion
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    criterion_smooth = utils.CrossEntropyLabelSmooth(1000, args.label_smooth)
    criterion_smooth = criterion_smooth.cuda()

    # load training data
    print('Loading data:')
    data_tmp = imagenet.Data(args)
    train_loader = data_tmp.train_loader
    val_loader = data_tmp.test_loader

    compress_rate = utils.get_cpr(args.compress_rate)

    # load baseline model
    logger.info('Loading baseline model:')
    ori_model = eval(args.arch)(compress_rate=[0.]*100, rank=0).cuda()
    ckpt = torch.load(args.ckpt)
    ori_model.load_state_dict(ckpt['state_dict'])

    # decompose
    logger.info('Decomposing model:')
    ori_model = decompose(ori_model, args.rank, args.n_iter_max,
                          args.n_iter_singular_error)

    # state dict after decomposing
    ori_state_dict = ori_model.state_dict()

    # prune
    logger.info('Pruning model:')
    model = eval(args.arch)(compress_rate=compress_rate, rank=args.rank).cuda()
    if args.arch == 'resnet_50':
        prune_resnet(model, ori_state_dict)

    logger.info(model)
    if len(args.gpu) > 1:
        device_id = []
        for i in range((len(args.gpu) + 1) // 2):
            device_id.append(i)
        model = nn.DataParallel(model, device_ids=device_id).cuda()

    # finetune
    logger.info('Finetuning model:')
    model = finetune(model, train_loader, val_loader,
                     args.epochs, criterion_smooth, criterion)

    # save model
    path = os.path.join(args.job_dir, f'{args.arch}_{name}.pt')
    torch.save({'state_dict': model.state_dict(),
                'rank': args.rank,
                'compress_rate': args.compress_rate},
               path)


def finetune(model, train_loader, val_loader, epochs, train_criterion, val_criterion):
    optimizer = torch.optim.SGD(model.parameters(
    ), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs-args.lr_warmup_epochs)
    warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.lr_warmup_epochs])

    _, best_top1_acc, _ = validate(val_loader, model, val_criterion, logger)
    best_model_state = copy.deepcopy(model.state_dict())
    epoch = 0
    while epoch < epochs:
        train(epoch, train_loader, model, train_criterion,
              optimizer, scheduler, logger)
        _, valid_top1_acc, valid_top5_acc = validate(
            val_loader, model, val_criterion, logger)

        if valid_top1_acc > best_top1_acc:
            best_top1_acc = valid_top1_acc
            best_model_state = copy.deepcopy(model.state_dict())

        cur_lr = optimizer.param_groups[0]['lr']
        wandb.log({'best_acc': max(valid_top1_acc, best_top1_acc),
                  'top1': valid_top1_acc, 'top5': valid_top5_acc, 'lr': cur_lr})

        epoch += 1
        logger.info('=>Best accuracy {:.3f}'.format(best_top1_acc))

    model.load_state_dict(best_model_state)

    return model


if __name__ == '__main__':
    main()
