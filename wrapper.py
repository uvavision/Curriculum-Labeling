import numpy as np
import scipy.misc
import urllib.request as urllib

import utils.dataloaders as dataloaders
from models.wideresnet import *
from models.lenet import *
from utils.helpers import *
import methods.entropy.curriculum_labeling as curriculum_labeling

import torch

class Wrapper:
    """
    All steps for our Curriculum Learning approach can be called from here.

    Args:
        args (dictionary): all user defined parameters
    """

    def __init__(self, args):
        """
        Initiazile the Model with all the parameters predifined by the user - check for the command_line_example.py file for all variables -
        All possible configurations should be explicitly defined and passed through a dictionary (args) 

        Args:
            args (dictionary): all user defined parameters
        """
        args.set_labeled_classes = [int(item) for item in args.set_labeled_classes.split(',')]
        args.set_unlabeled_classes = [int(item) for item in args.set_unlabeled_classes.split(',')]
        self.args = args
        self.model = None
        self.ema_model = None
        self.model_optimizer = None

    # #@property
    # def get_model(self):
    #     return self.model
    
    # #@property
    # def get_ema_model(self):
    #     return self.ema_model

    def create_network(self, ema=False):
        """
        Creates a model based on the parameter selection:
        - [WRN-28-2] was proposed by Oliver et. al. in "Realistic Evaluation of Deep Semi-Supervised Learning Algorithms" (https://arxiv.org/abs/1804.09170).
        - [CNN13] some papers still report top-1 test error using this architecture - Springenberg et. al. in "Striving for simplicity: The all convolutional net" (https://arxiv.org/abs/1412.6806).
        - [ResNet50] usually trained for ImageNet experiments - He et. al. in "Deep residual learning for image recognition" (https://arxiv.org/abs/1512.03385).

        Args:
            ema (bool, optional): if the model is a Teacher model or not. Defaults to False.
        """
        print('Build network -> ' + self.args.arch)
        print ('Dataset -> ' + self.args.dataset)
        print('Num classes ->', self.args.num_classes)
        if self.args.use_zca: 
            print('Use ZCA')

        if self.args.arch in ['cnn13','WRN28_2']:
            net = eval(self.args.arch)(self.args.num_classes, self.args.dropout)
        elif self.args.arch in ['resnet50']:
            import torchvision
            net = torchvision.models.resnet50(pretrained=False)
        else:
            assert False, "Error : Network should be cnn13, WRN28_2 or resnet50"

        if ema:
            for param in net.parameters():
                param.detach_()
            self.ema_model = net
        else:
            self.model = net

    def set_data(self, data):
        """
        Sets/updates data values to corresponding dictionary entry - executed after any dataset operation

        Args:
            data (array): dataset references
        """
        num_classes, train_data, train_data_noT, test_data = data
        # set dataset references
        self.args.num_classes = num_classes
        self.args.train_data = train_data
        self.args.train_data_noT = train_data_noT
        self.args.test_data = test_data

    def set_loaders(self, loaders):
        """
        Sets/updates data values to corresponding dictionary entry - executed after any dataset operation

        Args:
            loaders (array): subsets of dataloaders, samplers and indices
        """        
        trainloader, \
        validloader, \
        unlabelledloader, \
        train_sampler, \
        unlabelled_sampler, \
        indices_train, \
	    indices_unlabelled, \
	    train_index_order, \
        unlabeled_index_order = loaders
        # update loaders
        self.args.trainloader = trainloader
        self.args.validloader = validloader
        self.args.unlabelledloader = unlabelledloader
        self.args.train_sampler = train_sampler
        self.args.unlabelled_sampler = unlabelled_sampler
        self.args.indices_train = indices_train
        self.args.indices_unlabelled = indices_unlabelled
        self.args.train_index_order = train_index_order
        self.args.unlabeled_index_order = unlabeled_index_order

    def prepare_datasets(self):
        """
        Prepare datasets for training based on the predifined parameters

        1) Download precomputed zca components and mean for CIFAR10
        2) Load training and test raw sets (download if necessary)
        3) Get subsets for labeled, unlabeled and validation samples (based on seed)
        4) [Optional] Get test set if in debug mode
        """
        # download precomputed zca components and mean for CIFAR10
        urllib.urlretrieve("http://cs.virginia.edu/~pcascante/zca_components.npy", "zca_components.npy")
        urllib.urlretrieve("http://cs.virginia.edu/~pcascante/zca_mean.npy", "zca_mean.npy")

        # load data
        data = dataloaders.load_data_subsets(self.args.augPolicy, self.args.dataset, self.args.data_dir)
        self.set_data(data)

        # load zca for cifar10
        zca_components = np.load('zca_components.npy')
        zca_mean = np.load('zca_mean.npy')
        self.args.zca_components = zca_components
        self.args.zca_mean = zca_mean

        # get randomized set for training
        loaders = dataloaders.get_train_dataloaders(self.args.dataset, self.args.train_data, self.args.train_data_noT, self.args.batch_size, self.args.n_cpus, self.args.num_labeled, self.args.num_valid_samples, self.args.seed, self.args.set_labeled_classes, self.args.set_unlabeled_classes, ordered=False)
        self.set_loaders(loaders)

        # get test set if in debug mode and for final evaluation
        testloader = dataloaders.get_test_dataloader(self.args.test_data, self.args.batch_size, self.args.n_cpus)
        self.args.testloader = testloader


    def set_model_hyperparameters(self, ema=False):
        """
        Set model hyperparameters based on the user parameter selection

        1) Check CUDA availability
        2) Allow use of multiple GPUs

        Args:
            ema (bool, optional): if the model is a Teacher model or not. Defaults to False.
        """
        if self.args.doParallel:
            if ema:
                self.ema_model = torch.nn.DataParallel(self.ema_model)
            else:
                self.model = torch.nn.DataParallel(self.model)

        if torch.cuda.is_available():
            if ema:
                self.ema_model = self.ema_model.cuda()
            else:
                self.model = self.model.cuda()
            self.args.use_cuda = True
            # torch.backends.cudnn.benchmark = True # I personally prefer this one, but lets set deterministic True for the sake of reproducibility
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            self.args.use_cuda = False

    def set_model_optimizer(self):
        """
        Set model optimizer based on user parameter selection

        1) Set SGD or Adam optimizer
        2) Set SWA if set (check you have downloaded the library using: pip install torchcontrib)
        3) Print if: Use ZCA preprocessing (sometimes useful for CIFAR10) or debug mode is on or off 
           (to check the model on the test set without taking decisions based on it -- all decisions are taken based on the validation set)
        """
        if self.args.optimizer == 'sgd':
            prRed ('... SGD ...')
            optimizer = torch.optim.SGD(self.model.parameters(), self.args.lr,
                                                                      momentum=self.args.momentum,
                                                                      weight_decay=self.args.weight_decay,
                                                                      nesterov=self.args.nesterov)
        else:
            prRed ('... Adam optimizer ...')
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)

        if self.args.swa:
            prRed ('Using SWA!')
            from torchcontrib.optim import SWA
            optimizer = SWA(optimizer)
        
        self.model_optimizer = optimizer

        if self.args.use_zca:
            prPurple ('*Use ZCA preprocessing*')
        if self.args.debug:
            prPurple ('*Debug mode on*')

    def update_datasets(self, indices_for_rotation, ordered=False):
        """
        In the pseudo-labeling case, update the dataset: add the unlabeled samples with their corresponding pseudo annotations to the labeled set

        Args:
            indices_for_rotation (array): indices of all unlabeled samples that can will be added to the labeled dataset for training
        """
        if self.args.augPolicy == 2:
            data = dataloaders.load_data_subsets(self.args.augPolicy, self.args.dataset, self.args.data_dir)
            self.set_data(data)

        loaders = dataloaders.get_train_dataloaders(self.args.dataset, self.args.train_data, self.args.train_data_noT, self.args.batch_size, self.args.n_cpus, self.args.num_labeled, self.args.num_valid_samples, self.args.seed, self.args.set_labeled_classes, self.args.set_unlabeled_classes, ordered=ordered, indices_for_rotation = indices_for_rotation)
        self.set_loaders(loaders)
        
    # def order_for_query_datasets(self, indices_for_rotation):
    #     """
    #     In the pseudo-labeling case, order the dataset to evaluate all the unlabeled samples with the model trained in the previous rotation
    #     """
    #     trainloader, validloader, unlabelledloader, train_sampler, unlabelled_sampler, indices_train, indices_unlabelled, train_index_order, unlabeled_index_order  = get_train_dataloaders(self.args.dataset, train_data, train_data_noT, self.args.batch_size, self.args.n_cpus, self.args.num_labeled, self.args.num_valid_samples, self.args.seed, self.args.set_labeled_classes, self.args.set_unlabeled_classes, ordered=True)

    def train_cl(self):
        """
        Executes the Curriculum Learning standard algorithm.

        1) Train only on labeled data
        2) Use trained model to get max scores of unlabeled data
        3) Compute threshold (check percentiles_holder parameter) based on max scores -> long tail distribution
        4) Pseudo-label
        5) Train next iteration
        6) Do it until (almost) all dataset is covered
        """
        cl = curriculum_labeling.Curriculum_Labeling(self.args, self.model, self.model_optimizer)
        # train using only labeled subset
        cl.train_iteration()

        # based on trained model, pseudo-annotate and re-train
        iteration = 1
        while True:
            # evaluate unlabeled set: get max scores, compute percentile, pseudo-annotate
            self.update_datasets({}, ordered=True)
            # pass updated values to the method
            cl.update_args(self.args, None, None)
            image_indices_hard_label = cl.do_iteration(iteration)
            # reset network
            self.create_network()
            self.set_model_hyperparameters()
            self.set_model_optimizer()
            # update indices -- add pseudo-labeled samples to labeled set
            self.update_datasets(list(image_indices_hard_label.keys()))
            cl.update_args(self.args, self.model, self.model_optimizer, update_model=True)
            # re-train
            cl.train_iteration(iteration=iteration, image_indices_hard_label=image_indices_hard_label)
            # check until almost all or all dataset is pseudo-labeled - stop
            if self.args.percentiles_holder * iteration >= 100:
                prGreen ('All dataset used. Process finished.')
                break
            iteration += 1

    def eval_cl(self):
        """
        Execute the evaluation of Curriculum Learning. Goes over all iterations and select the best one based on the validation accuracy.
        """
        cl = curriculum_labeling.Curriculum_Labeling(self.args)
        cl.evaluate_all_iterations()



        