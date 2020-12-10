"""
You can use this file as a starting point to train your models.
"""

import datetime
import argparse
import wrapper as super_glue

parser = argparse.ArgumentParser(description='Curriculum Labeling Implementation')

parser.add_argument('--dataset', metavar='DATASET', default='cifar10', choices=['cifar10','svhn','imagenet'], 
                    help='dataset: cifar10, svhn or imagenet' )
parser.add_argument('--num_labeled', default=400, type=int, metavar='L',
                    help='number of labeled samples per class')
parser.add_argument('--num_valid_samples', default=500, type=int, metavar='V',
                    help='number of validation samples per class')

parser.add_argument('--arch', default='cnn13', type=str, help='either of cnn13, WRN28_2, resnet50')
parser.add_argument('--dropout', default=0.0, type=float, metavar='DO', help='dropout rate')
parser.add_argument('--optimizer', type = str, default = 'sgd',
                    help='optimizer we are going to use. can be either adam of sgd')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=100, type=int,
                    metavar='N', help='mini-batch size (default: 100)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='max learning rate')
parser.add_argument('--initial_lr', default=0.0, type=float,
                    metavar='LR', help='initial learning rate when using linear rampup')
parser.add_argument('--lr_rampup', default=0, type=int, metavar='EPOCHS',
                    help='length of learning rate rampup in the beginning')
parser.add_argument('--lr_rampdown_epochs', default=150, type=int, metavar='EPOCHS',
                    help='length of learning rate cosine rampdown (>= length of training): the epoch at which learning rate \
                    reaches to zero')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--nesterov', action='store_true',
                    help='use nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

parser.add_argument('--checkpoint_epochs', default=500, type=int,
                    metavar='EPOCHS', help='checkpoint frequency (by epoch)')
parser.add_argument('--print_freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--root_dir', type = str, default = 'experiments',
                        help='folder where results are to be stored')
parser.add_argument('--data_dir', type = str, default = '/data/cifar10/',
                        help='folder where data is stored')
parser.add_argument('--n_cpus', default=12, type=int,
                    help='number of cpus for data loading')

parser.add_argument('--add_name', type=str, default='SSL_Test')
parser.add_argument('--doParallel', dest='doParallel', action='store_true',
                    help='use DataParallel')

parser.add_argument('--use_zca', dest='use_zca', action='store_true',
                    help='use zca whitening')                    

parser.add_argument('--pretrainedEval', dest = 'pretrainedEval', action = 'store_true',
                    help = 'use pre-trained model')
parser.add_argument('--pretrainedFrom', default = '', type = str, metavar = 'PATH',
                    help = 'path to pretrained results (default: none)')
parser.add_argument('-e', '--evaluate', action='store_true',
                    help='evaluate model on evaluation set')
parser.add_argument('-evaluateLabeled', '--evaluateLabeled', action='store_true',
                    help='evaluate model on labeled set')
parser.add_argument('-getLabeledResults', '--getLabeledResults', action='store_true',
                    help='get the results of new model using labeled set')

parser.add_argument('--set_labeled_classes', dest = 'set_labeled_classes', default='0,1,2,3,4,5,6,7,8,9', type=str,
                    help='set the classes to treat as the label set')
parser.add_argument('--set_unlabeled_classes', dest = 'set_unlabeled_classes', default='0,1,2,3,4,5,6,7,8,9', type=str,
                    help='set the classes to treat as the unlabeled set')

parser.add_argument('--percentiles_holder', dest = 'percentiles_holder', default=20, type=int, help='mu parameter - sets the steping percentile for thresholding after each iteration')
parser.add_argument('--static_threshold', dest='static_threshold', action='store_true',
                    help='use static threshold')  

parser.add_argument('--seed', dest = 'seed', default=0, type=int, help='define seed for random distribution of dataset')

parser.add_argument('--augPolicy', default=2, type=int, help='augmentation policy: 0 for none, 1 for moderate, 2 for heavy (random-augment)')

parser.add_argument('--swa', action='store_true', help='Apply SWA')
parser.add_argument('--swa_start', type=int, default=200, help='Start SWA')
parser.add_argument('--swa_freq', type=float, default=5, help='Frequency')

parser.add_argument('--mixup', action='store_true', help='Apply Mixup to inputs')
parser.add_argument('--alpha', default=1., type=float, help='mixup interpolation coefficient (default: 1)')

parser.add_argument('--debug', action='store_true', help='Track the testing accuracy, only for debugging purposes')


if __name__ == '__main__':
    # parse arguments
    args = parser.parse_args()

    # create wrapper and prepare datasets
    wrapper = super_glue.Wrapper(args)
    wrapper.prepare_datasets()
    # create model, set hyperparameters and set optimizer (SGD or Adam)
    wrapper.create_network()
    wrapper.set_model_hyperparameters()
    wrapper.set_model_optimizer()
    ## uncomment to print the model
    # print (wrapper.model)

    # curriculum learning calls | train/evaluate
    # train cl
    wrapper.train_cl()
    # evaluate cl 
    wrapper.eval_cl()



    

    