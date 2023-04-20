import os
import datetime
import argparse
import wandb
import copy
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
import torch.utils.data
import tensorly as tl

import utils.common as utils
from data import cifar10
from models.cifar10.vgg_coring import vgg_16_bn

from decomposition.decomposition import cp_decomposition_conv_layer


def parse_args():
    parser = argparse.ArgumentParser("Cifar-10 decomposition")

    parser.add_argument('--data_dir', type=str,
                        default='data', help='path to dataset')
    parser.add_argument('--arch', type=str,
                        default='vgg_16_bn', help='architecture')
    parser.add_argument('--pruned', type=str, default='checkpoint/cifar10/pruned/vgg16_[0.05]*7+[0.2]*5.pt',
                        help='pruned model path')
    parser.add_argument('--job_dir', type=str, default='result',
                        help='path for saving trained models')
    parser.add_argument('--batch_size', type=int,
                        default=256, help='batch size')
    parser.add_argument('--epochs', type=int, default=400,
                        help='num of fine-tuning epochs')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='init learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float,
                        default=5e-3, help='weight decay')
    parser.add_argument('--div_factor', type=float, default=10,
                        help='div factor of OneCycle Learning rate Schedule (default: 10)')
    parser.add_argument('--final_div_factor', type=float, default=100,
                        help='final div factor of OneCycle Learning rate Schedule (default: 100)')
    parser.add_argument('--pct_start', type=float, default=0.1,
                        help='pct_start of OneCycle Learning rate Schedule (default: 0.1)')
    parser.add_argument('--gpu', type=str, default='0',
                        help='Select gpu to use')
    parser.add_argument("-r", "--rank", dest="rank", type=int, default=5,
                        help="use pre-specified rank for all layers")
    parser.add_argument('--dcp_epochs', type=int, default=10,
                        help='num of fine-tuning epochs after decomposing each layer')
    parser.add_argument('--dcp_max_epochs', type=int, default=50,
                        help='max num of fine-tuning epochs after decomposing each layer')
    parser.add_argument('--dcp_acc_drop', type=float, default=0.05,
                        help='accuracy drop threshold for decomposing each layer')
    parser.add_argument("-cpr", '--compress_rate', type=str, default='[0.]*100',
                        help='compress rate of each conv')

    return parser.parse_args()


args = parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

print_freq = (256*50)//args.batch_size

args.job_dir = os.path.join(args.job_dir, args.arch,
                            args.compress_rate, str(args.rank))
if not os.path.isdir(args.job_dir):
    os.makedirs(args.job_dir)

utils.record_config(args)
now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
logger = utils.get_logger(os.path.join(
    args.job_dir, 'logger_prune_decompose'+now+'.txt'))


def finetune(model, train_loader, val_loader, num_epochs, max_num_epochs, ori_acc, acc_drop_threshold):
    criterion = nn.CrossEntropyLoss()
    # use a small learning rate
    optimizer = torch.optim.SGD(model.parameters(
    ), lr=0.5*args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=int(max_num_epochs/2), gamma=0.1)

    _, best_top1_acc, _ = validate(val_loader, model, criterion)
    best_model_state = copy.deepcopy(model.state_dict())
    epoch = 0
    while epoch < max_num_epochs:
        train(epoch, train_loader, model, criterion, optimizer, scheduler)
        _, valid_top1_acc, _ = validate(val_loader, model, criterion)

        if valid_top1_acc > best_top1_acc:
            print(f'valid_top1_acc > best_top1_acc ',
                  valid_top1_acc, best_top1_acc)
            best_top1_acc = valid_top1_acc
            best_model_state = copy.deepcopy(model.state_dict())

        acc_drop = ori_acc - best_top1_acc
        if (acc_drop < acc_drop_threshold) and (epoch > num_epochs-1):
            break
        epoch += 1

    model.load_state_dict(best_model_state)

    return model, epoch


def main():
    # init wandb
    name = args.compress_rate + '_' + str(args.rank)
    wandb.init(name=name, project='TENDING_prune_decompose' +
               '_' + args.arch, config=vars(args))

    tl.set_backend("pytorch")

    # load training data
    train_loader, val_loader = cifar10.load_data(args)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    cudnn.benchmark = True
    cudnn.enabled = True
    logger.info("args = %s", args)

    # load baseline model
    logger.info('Loading pruned model')
    compress_rate = utils.get_cpr(args.compress_rate)
    model = vgg_16_bn(compress_rate=compress_rate).cuda()
    ckpt = torch.load(args.pruned, map_location='cuda:0')
    model.load_state_dict(ckpt['state_dict'], strict=False)
    # decompose
    _, ori_acc, _ = validate(val_loader, model, criterion)
    logger.info('Decomposing model:')
    for name, module in reversed(list(model.features._modules.items())):
        if isinstance(module, nn.Conv2d):
            logger.info(name)
            cpd_layer = cp_decomposition_conv_layer(module, args.rank)
            model.features._modules[name] = cpd_layer
            _, decomposed_acc, _ = validate(val_loader, model, criterion)
            model, epoch = finetune(model, train_loader, val_loader, num_epochs=args.dcp_epochs,
                                    max_num_epochs=args.dcp_max_epochs, ori_acc=ori_acc, acc_drop_threshold=args.dcp_acc_drop)
            _, finetuned_acc, _ = validate(val_loader, model, criterion)
            wandb.log({'layer': int(name[4:]), 'decomposed_acc': decomposed_acc,
                      'finetuned_acc': finetuned_acc, 'epoch_used': epoch})

    # inspect decomposed model
    logger.info('Evaluating decomposed model:')
    _, decomposed_acc, _ = validate(val_loader, model, criterion)
    wandb.log({'final_decomposed_acc': decomposed_acc})

    # fine-tuning
    logger.info('Finetuning model:')
    optimizer = torch.optim.SGD(model.parameters(
    ), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    start_epoch = 0
    best_top1_acc = 0

    # train the modelI
    epoch = start_epoch
    while epoch < args.epochs:
        train(epoch,  train_loader, model, criterion, optimizer, scheduler)
        _, valid_top1_acc, _ = validate(
            val_loader, model, criterion)

        is_best = False
        if valid_top1_acc > best_top1_acc:
            best_top1_acc = valid_top1_acc
            is_best = True

        utils.save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_top1_acc': best_top1_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, args.job_dir)

        cur_lr = optimizer.param_groups[0]["lr"]
        wandb.log({'best_acc': max(valid_top1_acc, best_top1_acc),
                  'top1': valid_top1_acc, 'lr': cur_lr})

        epoch += 1
        logger.info("=>Best accuracy {:.3f}".format(best_top1_acc))

    wandb.save(os.path.join(args.job_dir, '*'))


def train(epoch, train_loader, model, criterion, optimizer, scheduler):
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')

    model.train()

    # for param_group in optimizer.param_groups:
    #     cur_lr = param_group['lr']
    cur_lr = optimizer.param_groups[0]["lr"]
    logger.info('learning_rate: ' + str(cur_lr))

    num_iter = len(train_loader)
    for i, (images, target) in enumerate(train_loader):
        images = images.cuda()
        target = target.cuda()

        # compute outputy
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


if __name__ == '__main__':
    main()
