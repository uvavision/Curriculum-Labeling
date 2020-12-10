import re
import argparse
import os
import shutil
import time
import math
from itertools import repeat, cycle

import logging
from logger import Logger

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torchvision.datasets

from collections import OrderedDict
import sys
import pickle

# from mean_teacher import architectures, datasets, data, losses, ramps, cli
# from mean_teacher.run_context import RunContext
# from mean_teacher.data import NO_LABEL
from mean_teacher.utils import *
from mean_teacher import ramps

from utils import *
#from utilsOutClasses import *

from networks.wide_resnet import *
from networks.lenet import *
#from networks.shake_shake_hysts import Network as FE

import multiprocessing
from scipy.spatial import distance

#from CustomImagenet32 import *


parser = argparse.ArgumentParser(description='Self-training code')
parser.add_argument('--dataset', metavar='DATASET', default='cifar10',
                        choices=['cifar10','svhn','imagenet'],
                        help='dataset: cifar10, svhn or imagenet' )
parser.add_argument('--num_labeled', default=400, type=int, metavar='L',
                    help='number of labeled samples per class')
parser.add_argument('--num_valid_samples', default=500, type=int, metavar='V',
                    help='number of validation samples per class')
parser.add_argument('--arch', default='cnn13', type=str, help='either of cnn13, WRN28_2, resnet50')
parser.add_argument('--dropout', default=0.0, type=float,
                    metavar='DO', help='dropout rate')

parser.add_argument('--optimizer', type = str, default = 'sgd',
                        help='optimizer we are going to use. can be either adam of sgd')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=100, type=int,
                    metavar='N', help='mini-batch size (default: 100)')

parser.add_argument('--root_dir', type = str, default = 'experiments',
                        help='folder where results are to be stored')
parser.add_argument('--data_dir', type = str, default = 'data/cifar10/',
                        help='folder where data is stored')
parser.add_argument('--n_cpus', default=12, type=int,
                    help='number of cpus for data loading')

parser.add_argument('--add_name', type=str, default='GoodDistribution_NoNN')
parser.add_argument('--doParallel', dest='doParallel', action='store_true',
                    help='use DataParallel')

parser.add_argument('--useZCA', dest='useZCA', action='store_true',
                    help='use zca whitening')                    

parser.add_argument('--set_labeled_classes', dest = 'set_labeled_classes', default='0,1,2,3,4,5,6,7,8,9', type=str,
                    help='set the classes to treat as the label set')
parser.add_argument('--set_unlabeled_classes', dest = 'set_unlabeled_classes', default='0,1,2,3,4,5,6,7,8,9', type=str,
                    help='set the classes to treat as the unlabeled set')

parser.add_argument('--rotations', dest = 'rotations', default='0-Rotation,1-Rotation,2-Rotation,3-Rotation,4-Rotation,5-Rotation', type=str,
                    help='query all iterations (total of 5 rotations)')

parser.add_argument('--seed', dest = 'seed', default=0, type=int,
                    help='define seed for random distribution of dataset')

parser.add_argument('--augPolicy', default=2, type=int, help='augmentation policy: 0 for none, 1 for moderate, 2 for heavy (autoaugment)')

### get net####
def getNetwork(args, num_classes):

    if args.arch in ['cnn13','WRN28_2']:
        net = eval(args.arch)(num_classes, args.dropout)
    elif args.arch in ['resnet50']:
        net = torchvision.models.resnet50(pretrained=False)
    else:
        prRed('Error : Network should be cnn13, WRN28_2 or resnet50')
        sys.exit(0)

    return net

args = parser.parse_args()
use_cuda = torch.cuda.is_available()

def main():
    global num_classes
    global zca_components
    global zca_mean
    global exp_dir

    #### load data###
    # if args.dataset == 'cifar10':
    #     data_source_dir = args.data_dir
    #     trainloader, validloader, unlabelledloader, testloader, num_classes, train_sampler, unlabelled_sampler, indices_train, indices_unlabelled, trainIndexOrder, unlabeledIndexOrder, train_data = load_data_subset(args.augPolicy, args.batch_size, args.n_cpus ,'cifar10', data_source_dir, labels_per_class = args.num_labeled, valid_labels_per_class = args.num_valid_samples, seed = args.seed)
    #     zca_components = np.load('zca_components.npy')
    #     zca_mean = np.load('zca_mean.npy')
    # if args.dataset == 'svhn':
    #     data_source_dir = args.data_dir
    #     trainloader, validloader, unlabelledloader, testloader, num_classes, train_sampler, unlabelled_sampler, indices_train, indices_unlabelled, trainIndexOrder, unlabeledIndexOrder, train_data = load_data_subset(args.augPolicy, args.batch_size, args.n_cpus ,'svhn', data_source_dir, labels_per_class = args.num_labeled, valid_labels_per_class = args.num_valid_samples, seed = args.seed)
    # if args.dataset == 'imagenet':
    #     data_source_dir = args.data_dir
    #     trainloader, validloader, unlabelledloader, testloader, num_classes, train_sampler, unlabelled_sampler, indices_train, indices_unlabelled, trainIndexOrder, unlabeledIndexOrder, train_data = load_data_subset(args.augPolicy, args.batch_size, args.n_cpus ,'imagenet', data_source_dir, labels_per_class = args.num_labeled, valid_labels_per_class = args.num_valid_samples, seed = args.seed)

    args.set_labeled_classes = [int(item) for item in args.set_labeled_classes.split(',')]
    args.set_unlabeled_classes = [int(item) for item in args.set_unlabeled_classes.split(',')]

    #### load data###
    num_classes, train_data, train_data_noT, test_data = load_data_subsets(args.augPolicy, args.dataset, args.data_dir)
    zca_components = None #np.load('zca_components.npy')
    zca_mean = None #np.load('zca_mean.npy')

    # get set for validation
    #trainloader, validloader, unlabelledloader, train_sampler, unlabelled_sampler, indices_train, indices_unlabelled, trainIndexOrder, unlabeledIndexOrder  = get_train_dataloaders(args.dataset, train_data, train_data_noT, args.batch_size, args.n_cpus, args.num_labeled, args.num_valid_samples, args.seed, ordered=False)
    trainloader, validloader, unlabelledloader, train_sampler, unlabelled_sampler, indices_train, indices_unlabelled, trainIndexOrder, unlabeledIndexOrder  = get_train_dataloaders(args.dataset, train_data, train_data_noT, args.batch_size, args.n_cpus, args.num_labeled, args.num_valid_samples, args.seed, args.set_labeled_classes, args.set_unlabeled_classes, ordered=True)
    testloader = get_test_dataloader(test_data, args.batch_size, args.n_cpus)

    args.rotations = [item for item in args.rotations.split(',')]

    print('Build network -> ' + args.arch)
    print ('Dataset -> ' + args.dataset)
    print('-- Use ZCA', args.useZCA)
    print('Num classes', num_classes)
    queryModel = getNetwork(args, num_classes)

    if args.doParallel:
        queryModel = torch.nn.DataParallel(queryModel)

    if use_cuda:
        queryModel = queryModel.cuda()
        cudnn.benchmark = True

    #exp_dir = os.path.join(args.root_dir, '{}/{}/{}'.format(args.dataset, args.arch, args.add_name))
    exp_dir = args.root_dir
    prGreen('Results will be saved to this folder: {}'.format(exp_dir))
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    for r in args.rotations:
        print("=> loading pretrained checkpoint '{}'".format(exp_dir + '/{}.best.ckpt'.format(r)))
        checkpoint = torch.load(exp_dir  + '/{}.best.ckpt'.format(r))
        best_prec1 = checkpoint['best_prec1']
        queryModel.load_state_dict(checkpoint['state_dict'])
        print("=> loaded pretrained checkpoint '{}' (epoch {})".format(exp_dir + '/{}.best.ckpt'.format(r), checkpoint['epoch']))

        prGreen("Val: Evaluating the model:")
        validate(validloader, queryModel, args.start_epoch)
        print ('=====================================================')
        prGreen("Test: Evaluating the model:")
        validate(testloader, queryModel, args.start_epoch)

def parse_dict_args(**kwargs):
    global args

    def to_cmdline_kwarg(key, value):
        if len(key) == 1:
            key = "-{}".format(key)
        else:
            key = "--{}".format(re.sub(r"_", "-", key))
        value = str(value)
        return key, value

    kwargs_pairs = (to_cmdline_kwarg(key, value)
                    for key, value in kwargs.items())
    cmdline_args = list(sum(kwargs_pairs, ()))
    args = parser.parse_args(cmdline_args)

#convert targets [2,3,4,5,6,7] to [0,1,2,3,4,5] for only animal classes
def castClasses(targets):
    castedTargets = torch.zeros(len(targets), dtype=torch.long)
    for i, t in enumerate(targets):
        castedTargets[i] = t-2
    return castedTargets

def validate(eval_loader, model, epoch, testing = False, useZCA = True, k=5):
    class_criterion = nn.CrossEntropyLoss().cuda() #reduction='sum', ignore_index=NO_LABEL
    meters = AverageMeterSet()

    # switch to evaluate mode
    model.eval()

    true_labels = []
    predicted_labels = []
    softmax_res = []

    end = time.time()
    for i, (input, target) in enumerate(eval_loader):
        meters.update('data_time', time.time() - end)

        if args.dataset == 'cifar10':
            if args.useZCA:
                input = apply_zca(input, zca_mean, zca_components)

        with torch.no_grad():
            input_var = torch.autograd.Variable(input.cuda())
        with torch.no_grad():
            target_var = torch.autograd.Variable(target.cuda(non_blocking = True))

        minibatch_size = len(target_var)

        # compute output
        output1 = model(input_var)
        softmax1 = F.softmax(output1, dim=1)
        class_loss = class_criterion(output1, target_var) #/ minibatch_size

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output1.data, target_var.data, topk=(1, k))
        meters.update('class_loss', class_loss.item(), minibatch_size)
        meters.update('top1', prec1[0], minibatch_size)
        meters.update('error1', 100.0 - prec1[0], minibatch_size)
        meters.update('top5', prec5[0], minibatch_size)
        meters.update('error5', 100.0 - prec5[0], minibatch_size)

        # measure elapsed time
        meters.update('batch_time', time.time() - end)
        end = time.time()

        #print (softmax1.cpu().detach().numpy())
        #exit()
        for i in range (len(softmax1)):
            predicted_labels.append(np.argmax(softmax1.cpu().detach().numpy()[i]))
        true_labels.extend(target.cpu().detach().numpy())
        softmax_res.extend(softmax1.cpu().detach().numpy())

    pickle.dump([true_labels, predicted_labels, softmax_res], open('Full_Results.p', "wb"))

    print(' * Prec@1 {top1.avg:.3f}\tPrec@{k} {top5.avg:.3f}'
          .format(top1=meters['top1'], k=k, top5=meters['top5']))

    return meters['top1'].avg, meters['top5'].avg, meters['class_loss'].avg


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    #labeled_minibatch_size = max(target.ne(NO_LABEL).sum(), 1e-8).type(torch.cuda.FloatTensor)
    minibatch_size = len(target)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / minibatch_size))
    return res


if __name__ == '__main__':
     main()
