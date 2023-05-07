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
from data import cifar10
from models.cifar10.vgg import vgg_16_bn
from models.cifar10.resnet import resnet_56
from decomposition.decomposition import decompose


def parse_args():
    parser = argparse.ArgumentParser('Cifar-10 decomposition')

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
    parser.add_argument('-r', '--rank', dest='rank', type=int, default=6,
                        help='use pre-specified rank for all layers')
    parser.add_argument('-cpr', '--compress_rate', type=str, default='[0.]*100',
                        help='list of compress rate of each layer')
    parser.add_argument('--n_iter_max', type=int, default=300,
                        help='max number of iterations for parafac')
    parser.add_argument('--n_iter_singular_error', type=int, default=3,
                        help='number of iterations for singular maxtrix error handler')
    parser.add_argument('--name', type=str, default='',
                        help='wandb project name')

    return parser.parse_args()


args = parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

print_freq = (256*50)//args.batch_size

args.job_dir = os.path.join(args.job_dir, args.arch,
                            str(args.rank), args.compress_rate)
if not os.path.isdir(args.job_dir):
    os.makedirs(args.job_dir)

utils.record_config(args)
now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
logger = utils.get_logger(os.path.join(args.job_dir, now+'.txt'))


def main():
    logger.info('args = %s', args)
    name = f'{args.compress_rate}_{args.rank}'
    wandb.init(name=name,
               project=f'NORTON_Decompose_{args.name}_{args.arch}',
               config=vars(args))

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
    model = eval(args.arch)(compress_rate=compress_rate).cuda()
    ckpt = torch.load(args.ckpt, map_location='cuda:0')
    model.load_state_dict(ckpt['state_dict'])

    # decompose
    logger.info('Decomposing model:')
    model = decompose(model, args.rank,
                      args.n_iter_max, args.n_iter_singular_error)
    _, dcp_acc, _ = validate(val_loader, model, criterion)
    wandb.log({'decomposed_acc': dcp_acc})

    # finetune
    logger.info('Finetuning model:')
    model = finetune(model, train_loader, val_loader, args.epochs)

    # save model
    path = os.path.join(args.job_dir, f'{args.arch}_{name}.pt')
    torch.save({'state_dict': model.state_dict(),
                'rank': args.rank},
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
