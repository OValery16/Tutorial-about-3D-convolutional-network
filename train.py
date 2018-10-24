import argparse
import os
import sys
import shutil
import json
import glob
import signal
import pickle
import tensorboardX
import torch
import torch.nn as nn
import numpy as np
import csv

from data_loader import VideoFolder,VideoFolder_test
from model import ConvColumn5,ConvColumn6
from torchvision.transforms import *

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

str2bool = lambda x: (str(x).lower() == 'true')

parser = argparse.ArgumentParser(
    description='PyTorch Jester Training using JPEG')
parser.add_argument('--config', '-c', help='json config file path')
parser.add_argument('--eval_only', '-e', default=False, type=str2bool,
                    help="evaluate trained model on validation data.")
parser.add_argument('--test_only', '-t', default=False, type=str2bool,
                    help="test the trained model on the test set.")
parser.add_argument('--resume', '-r', default=False, type=str2bool,
                    help="resume training from given checkpoint.")
parser.add_argument('--use_gpu', default=True, type=str2bool,
                    help="flag to use gpu or not.")
parser.add_argument('--gpus', '-g', help="gpu ids for use.")

args = parser.parse_args()
if len(sys.argv) < 2:
    parser.print_help()
    sys.exit(1)

device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")

if args.use_gpu:
    gpus = [int(i) for i in args.gpus.split(',')]
    print("=> active GPUs: {}".format(args.gpus))

best_prec1 = 0

# load config file
with open(args.config) as data_file:
    config = json.load(data_file)


def main():
    global args, best_prec1

    # set run output folder
    model_name = config["model_name"]
    output_dir = config["output_dir"]
    print("=> Output folder for this run -- {}".format(model_name))
    save_dir = os.path.join(output_dir, model_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.makedirs(os.path.join(save_dir, 'plots'))

    # adds a handler for Ctrl+C
    def signal_handler(signal, frame):
        """
        Remove the output dir, if you exit with Ctrl+C and
        if there are less then 3 files.
        It prevents the noise of experimental runs.
        """
        num_files = len(glob.glob(save_dir + "/*"))
        if num_files < 1:
            shutil.rmtree(save_dir)
        print('You pressed Ctrl+C!')
        sys.exit(0)
    # assign Ctrl+C signal handler
    signal.signal(signal.SIGINT, signal_handler)

    # create model

    model = ConvColumn6(config['num_classes'])

    # multi GPU setting
    if args.use_gpu:
        model = torch.nn.DataParallel(model, device_ids=gpus).to(device)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(config['checkpoint']):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(config['checkpoint'])
            args.start_epoch = 0#checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            for key, value in checkpoint['state_dict'].items() :
                print (key)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(config['checkpoint'], checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(
                config['checkpoint']))

    transform_train = Compose([  
        RandomAffine(degrees=[-10, 10], translate=[0.15, 0.15]),
        CenterCrop(84),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
    ])
    transform_valid = Compose([
        CenterCrop(84),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
    ])

    train_data = VideoFolder(root=config['train_data_folder'],
                             csv_file_input=config['train_data_csv'],
                             csv_file_labels=config['labels_csv'],
                             clip_size=config['clip_size'],
                             nclips=1,
                             step_size=config['step_size'],
                             is_val=False,
                             transform=transform_train,
                             )

    print(" > Using {} processes for data loader.".format(
        config["num_workers"]))
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=config['batch_size'], shuffle=True,
        num_workers=config['num_workers'], pin_memory=True,
        drop_last=True)

    val_data = VideoFolder(root=config['val_data_folder'],
                           csv_file_input=config['val_data_csv'],
                           csv_file_labels=config['labels_csv'],
                           clip_size=config['clip_size'],
                           nclips=1,
                           step_size=config['step_size'],
                           is_val=True,
                           transform=transform_valid,
                           )

    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=config['batch_size'], shuffle=False,
        num_workers=config['num_workers'], pin_memory=True,
        drop_last=False)

    assert len(train_data.classes) == config["num_classes"]

    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss().to(device)

    # define optimizer
    lr = config["lr"]
    last_lr = config["last_lr"]
    momentum = config['momentum']
    weight_decay = config['weight_decay']
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,amsgrad =True) 


    if args.eval_only:
        validate(val_loader, model, criterion, train_data.classes_dict)
        return

    # set end condition by num epochs
    num_epochs = int(config["num_epochs"])
    if num_epochs == -1:
        num_epochs = 999999

    print(" > Training is getting started...")
    print(" > Training takes {} epochs.".format(num_epochs))
    start_epoch = args.start_epoch if args.resume else 0
    train_writer = tensorboardX.SummaryWriter("logs")
    for epoch in range(start_epoch, num_epochs):


        # train for one epoch
        train_loss, train_top1, train_top5 = train(
            train_loader, model, criterion, optimizer, epoch)
        # evaluate on validation set
        val_loss, val_top1, val_top5 = validate(val_loader, model, criterion)
		
		# write in tensorboard
        train_writer.add_scalar('loss', train_loss, epoch + 1)
        train_writer.add_scalar('top1', train_top1, epoch + 1)
        train_writer.add_scalar('top5', train_top5, epoch + 1)

        train_writer.add_scalar('val_loss', val_loss, epoch + 1)
        train_writer.add_scalar('val_top1', val_top1, epoch + 1)
        train_writer.add_scalar('val_top5', val_top5, epoch + 1)

        # remember best prec@1 and save checkpoint
        is_best = val_top1 > best_prec1
        best_prec1 = max(val_top1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': "Conv4Col",
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, config)


def train(train_loader, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    for i, (input, target) in enumerate(train_loader):

        input, target = input.to(device), target.to(device)

        model.zero_grad()

        # compute output and loss
        output,_ = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.detach(), target.detach().cpu(), topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % config["print_freq"] == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                      epoch, i, len(train_loader), loss=losses, top1=top1, top5=top5))
    return losses.avg, top1.avg, top5.avg


def validate(val_loader, model, criterion, class_to_idx=None):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    logits_matrix = []
    targets_list = []

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):

            input, target = input.to(device), target.to(device)

            # compute output and loss
            output,_ = model(input)
            loss = criterion(output, target)

            if args.eval_only:
                logits_matrix.append(output.detach().cpu().numpy())
                targets_list.append(target.detach().cpu().numpy())

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.detach(), target.detach().cpu(), topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            if i % config["print_freq"] == 0:
                print('Test: [{0}/{1}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                          i, len(val_loader), loss=losses, top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

        if args.eval_only:
            logits_matrix = np.concatenate(logits_matrix)
            targets_list = np.concatenate(targets_list)
            print(logits_matrix.shape, targets_list.shape)
            print(targets_list)
            save_results(logits_matrix, targets_list, class_to_idx, config)
        return losses.avg, top1.avg, top5.avg


def save_results(logits_matrix, targets_list, class_to_idx, config):
    print("Saving inference results ...")
    path_to_save = os.path.join(
        config['output_dir'], config['model_name'], "test_results.pkl")
    with open(path_to_save, "wb") as f:
        pickle.dump([logits_matrix, targets_list, class_to_idx], f)
        


def save_checkpoint(state, is_best, config, filename='checkpoint.pth.tar'):
    checkpoint_path = os.path.join(
        config['output_dir'], config['model_name'], filename)
    model_path = os.path.join(
        config['output_dir'], config['model_name'], 'model_best.pth.tar')
    torch.save(state, checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path, model_path)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.cpu().topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
    #trainEnsemble()