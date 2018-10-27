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
from callbacks import PlotLearning, MonitorLRDecay, AverageMeter
from model import Classifier,ConvColumn5,ConvColumn6,ConvColumn7,ConvColumn8,ConvColumn9
from torchvision.transforms import *

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

def save_results(logits_matrix, targets_list, class_to_idx, config):
    print("Saving inference results ...")
    path_to_save = os.path.join(
        config['output_dir'], config['model_name'], "test_results.pkl")
    with open(path_to_save, "wb") as f:
        pickle.dump([logits_matrix, targets_list, class_to_idx], f)
        
    '''
    path_to_save2 = os.path.join(
        config['output_dir'], config['model_name'], "test_results.csv")
    with open(path_to_save2, mode='w') as employee_file:
        employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        employee_writer.writerow(['John Smith', 'Accounting', 'November'])
        employee_writer.writerow(['Erica Meyers', 'IT', 'March'])
    '''

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

def trainEnsemble():
    global args, best_prec1

    # set run output folder
    model_name = "classifier"
    output_dir = config["output_dir"]
    
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
    #model = ConvColumn(config['num_classes'])
    
    model0 = ConvColumn6(config['num_classes'])
    model0 = torch.nn.DataParallel(model0, device_ids=gpus).to(device)

    if os.path.isfile("trainings/jpeg_model/jester_conv6/checkpoint.pth.tar"):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load("trainings/jpeg_model/jester_conv6/checkpoint.pth.tar")
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model0.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format("trainings/jpeg_model/jester_conv6/checkpoint.pth.tar", checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(
            config['checkpoint']))    
    
    
    model1 = ConvColumn7(config['num_classes'])
    model1 = torch.nn.DataParallel(model1, device_ids=gpus).to(device)


    if os.path.isfile("trainings/jpeg_model/jester_conv7/model_best.pth.tar"):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load("trainings/jpeg_model/jester_conv7/model_best.pth.tar")
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model1.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format("trainings/jpeg_model/jester_conv7/model_best.pth.tar", checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(
            config['checkpoint']))
       


    classifier=Classifier(config['num_classes'])
    classifier = torch.nn.DataParallel(classifier, device_ids=gpus).to(device)
    
    if os.path.isfile("trainings/jpeg_model/classifier/model_best.pth.tar"):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load("trainings/jpeg_model/classifier/model_best.pth.tar")
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        classifier.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format("trainings/jpeg_model/classifier/model_best.pth.tar", checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(
            config['checkpoint']))
    
    model3 = ConvColumn9(config['num_classes'])
    model3 = torch.nn.DataParallel(model3, device_ids=gpus).to(device)
    
    if os.path.isfile("trainings/jpeg_model/jester_conv9/model_best.pth.tar"):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load("trainings/jpeg_model/jester_conv9/model_best.pth.tar")
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model3.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format("trainings/jpeg_model/jester_conv9/model_best.pth.tar", checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(
            config['checkpoint']))
        
    model2 = ConvColumn8(config['num_classes'])
    model2 = torch.nn.DataParallel(model2, device_ids=gpus).to(device)
    
    if os.path.isfile("trainings/jpeg_model/jester_conv8/model_best.pth.tar"):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load("trainings/jpeg_model/jester_conv8/model_best.pth.tar")
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model2.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format("trainings/jpeg_model/jester_conv8/model_best.pth.tar", checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(
            config['checkpoint']))
        
    model4 = ConvColumn5(config['num_classes'])
    model4 = torch.nn.DataParallel(model4, device_ids=gpus).to(device)
    
    if os.path.isfile("trainings/jpeg_model/ConvColumn5/model_best.pth.tar"):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load("trainings/jpeg_model/ConvColumn5/model_best.pth.tar")
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model4.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format("trainings/jpeg_model/ConvColumn5/model_best.pth.tar", checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(
            config['checkpoint']))
    
    transform_train = Compose([ 
        RandomAffine(degrees=[-10, 10], translate=[0.15, 0.15],scale=[0.9, 1.1],shear=[-5, 5]),
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
    
    list_id_files=[]
    for i in val_data.csv_data:
        list_id_files.append(i.path[16:])
    print(len(list_id_files))
    
    ###########
    
   

    
    assert len(train_data.classes) == config["num_classes"]
    
    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss().to(device)

    # define optimizer
    lr = config["lr"]
    last_lr = config["last_lr"]
    momentum = config['momentum']
    weight_decay = config['weight_decay']
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr,amsgrad =True) 
    
    #torch.optim.SGD(classifier.parameters(), lr,
                                #momentum=momentum,
                                #weight_decay=weight_decay)

    # set callbacks
    plotter = PlotLearning(os.path.join(
        save_dir, "plots"), config["num_classes"])
    lr_decayer = MonitorLRDecay(0.6, 3)
    val_loss = 9999999

    # set end condition by num epochs
    num_epochs = int(config["num_epochs"])
    if num_epochs == -1:
        num_epochs = 999999
        
    if args.test_only:
        print("test")
        test_data = VideoFolder_test(root=config['val_data_folder'],
                           csv_file_input=config['test_data_csv'],
                           clip_size=config['clip_size'],
                           nclips=1,
                           step_size=config['step_size'],
                           is_val=True,
                           transform=transform_valid,
                           )
        
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=config['batch_size'], shuffle=False,
            num_workers=config['num_workers'], pin_memory=True,
            drop_last=False)

        list_id_files_test=[]
        for i in test_data.csv_data:
            list_id_files_test.append(i.path[16:])
        print(len(list_id_files_test))
        test_ensemble(test_loader, classifier,model1,model2,model3,list_id_files_test, criterion, train_data.classes_dict)
        return

    if args.eval_only:
        val_loss, val_top1, val_top5 = validate_ensemble(val_loader, classifier,model1,model2,model3,list_id_files, criterion, train_data.classes_dict)
        return

    # switch to evaluate mode
    model0.eval()
    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()
    classifier.train()

    logits_matrix = []
    targets_list = []
    
    new_input = np.array([])
    train_writer = tensorboardX.SummaryWriter("logs")
    
    for epoch in range(0, num_epochs):
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        lr = lr_decayer(val_loss, lr)
        print(" > Current LR : {}".format(lr))

        if lr < last_lr and last_lr > 0:
            print(" > Training is done by reaching the last learning rate {}".
                  format(last_lr))
            sys.exit(1)
        for i, (input, target) in enumerate(train_loader):
            input, target = input.to(device), target.to(device)
            
            with torch.no_grad():




                # compute output and loss
                output0,feature0 = model0(input)
                output1,feature1 = model1(input)
                output2,feature2 = model2(input)
                output3,feature3 = model3(input)
                output4,feature4 = model4(input)
                #sav=torch.cat((feature0,feature1,feature2,feature3,feature4),1)
                sav=torch.cat((output0,output1,output2,output3,output4),1)
            classifier.zero_grad()
            class_video=classifier(sav)
            loss = criterion(class_video, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(class_video.detach(), target.detach().cpu(), topk=(1, 5))
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
                          0, i, len(train_loader), loss=losses, top1=top1, top5=top5))


        val_loss, val_top1, val_top5 = validate_ensemble(val_loader, classifier,model0,model1,model2,model3,model4,list_id_files, criterion)

        train_writer.add_scalar('loss', loss, losses.avg)
        train_writer.add_scalar('top1', top1.avg, epoch + 1)
        train_writer.add_scalar('top5', top5.avg, epoch + 1)
        
        train_writer.add_scalar('val_loss', val_loss, epoch + 1)
        train_writer.add_scalar('val_top1', val_top1, epoch + 1)
        train_writer.add_scalar('val_top5', val_top5, epoch + 1)
        
        # remember best prec@1 and save checkpoint
        is_best = val_top1 > best_prec1
        best_prec1 = max(val_top1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': "Classifier",
            'state_dict': classifier.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, config)
    
def validate_ensemble(val_loader, classifier,model0,model1,model2,model3,model4,list_id_files, criterion, class_to_idx=None):

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model0.eval()
    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()
    classifier.eval()

    logits_matrix = []
    targets_list = []
    label_list = []
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):

            input, target = input.to(device), target.to(device)

            # compute output and loss
            output0,feature0 = model0(input)
            output1,feature1 = model1(input)
            output2,feature2 = model2(input)
            output3,feature3 = model3(input)
            output4,feature4 = model4(input)
            #sav=torch.cat((feature0,feature1,feature2,feature3,feature4),1)
            sav=torch.cat((output0,output1,output2,output3,output4),1)
            class_video=classifier(sav)
            loss = criterion(class_video, target)
            if args.eval_only:
                logits_matrix.append(class_video.detach().cpu().numpy())
                targets_list.append(target.detach().cpu().numpy())
                _, predicted = torch.max(class_video.data, 1)
                label_list.append(predicted.detach().cpu().numpy())
                total += target.size(0)
                correct += (predicted == target).sum()

            # measure accuracy and record loss
            prec1, prec5 = accuracy(class_video.detach(), target.detach().cpu(), topk=(1, 5))
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
            label_list = np.concatenate(label_list)
            print('Accuracy of the model: %d %%' % (100 * correct / total))
            print('Accuracy2 of the model: %d %%' % (100 * ((label_list == targets_list).sum()) / total))
            

            path_to_save2 = os.path.join(
                config['output_dir'], config['model_name'], "test_results.csv")
            with open(path_to_save2, mode='w') as csv_file:
                my_csv_writer = csv.writer(csv_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                for i in range(len(list_id_files)):
                    my_csv_writer.writerow([list_id_files[i], class_to_idx[label_list[i]]])

           
            print(logits_matrix.shape, targets_list.shape)
            print(class_to_idx)
            save_results(logits_matrix, targets_list, class_to_idx, config)
            
        return losses.avg, top1.avg, top5.avg
    
def test_ensemble(test_loader, classifier,model1,model2,model3,list_id_files, criterion, class_to_idx=None):

    model1.eval()
    model2.eval()
    model3.eval()
    classifier.eval()

    label_list = []

    with torch.no_grad():
        for i, (input,_) in enumerate(test_loader):

            input = input.to(device)

            # compute output and loss
            #output0,feature0 = model0(input)
            output1,feature1 = model1(input)
            output2,feature2 = model2(input)
            output3,feature3 = model3(input)
            sav=torch.cat((output1,output2,output3),1)
            class_video=classifier(sav)
            _, predicted = torch.max(class_video.data, 1)
            label_list.append(predicted.detach().cpu().numpy())


            
            if i % config["print_freq"] == 0:
                print('Test: [{0}/{1}]\t ======>In process)'.format(i, len(test_loader)))


        print(i)
        label_list = np.concatenate(label_list)
        path_to_save2 = os.path.join(
            config['output_dir'], config['model_name'], "testset_results.csv")
        with open(path_to_save2, mode='w') as csv_file:
            my_csv_writer = csv.writer(csv_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            print(len(label_list))
            print(len(list_id_files))
            for i in range(len(list_id_files)):
                my_csv_writer.writerow([list_id_files[i], class_to_idx[label_list[i]]])


if __name__ == '__main__':
    #main()
    trainEnsemble()