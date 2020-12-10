import os
import numpy as np
import time
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.logger import Logger
from utils.helpers import *
from utils.scheduler_ramps import *
from ..base import *

# from https://github.com/pytorch/contrib = pip install torchcontrib
from torchcontrib.optim import SWA

class Curriculum_Labeling(Train_Base):
    """
    Curriculum Labeling, method proposed by Cascante-Bonilla et. al. in https://arxiv.org/abs/2001.06001.
    """
    def __init__(self, args, model, model_optimizer):
        """
        Initialize the Curriculum Learning class with all methods and required variables
        This class use the model, optimizer, dataloaders and all the user parameters to train the CL algorithm proposed by Cascante-Bonilla et. al. in Curriculum Learning: (https://arxiv.org/abs/2001.06001)

        Args:
            args (dictionary): all user defined parameters with some pre-initialized objects (e.g., model, optimizer, dataloaders)
        """
        self.best_prec1 = 0

        ### list error and losses ###
        self.train_class_loss_list = []
        self.train_error_list = []
        self.train_lr_list = []
        self.val_class_loss_list = []
        self.val_error_list = []

        exp_dir = os.path.join(args.root_dir, '{}/{}/{}'.format(args.dataset, args.arch, args.add_name))
        prGreen('Results will be saved to this folder: {}'.format(exp_dir))

        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)

        self.args = args
        self.args.exp_dir = exp_dir
        self.model = model
        self.model_optimizer = model_optimizer

        # add TF Logger
        self.train_logger = Logger(os.path.join(self.args.exp_dir, 'TFLogs/train'))
        self.val_logger = Logger(os.path.join(self.args.exp_dir, 'TFLogs/val'))

    # mixup code from: https://arxiv.org/pdf/1710.09412.pdf ==> https://github.com/facebookresearch/mixup-cifar10
    def mixup_data(self, x, y, alpha=1.0, use_cuda=True):
        """
        Data augmentation technique proposed by Zhang et. al. that consists on interpolating two samples and their corresponding labels (https://arxiv.org/pdf/1710.09412.pdf).

        Args:
            x: input batch
            y: target batch
            alpha (float, optional): ~ Beta(alpha, alpha). Defaults to 1.0.
            use_cuda (bool, optional): if cuda is available. Defaults to True.

        Returns:
            Mixed inputs, pairs of targets, and lambda
        """
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        if use_cuda:
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        """
        Loss function to compute when mixup is applied.

        Args:
            criterion: loss function (e.g. categorical crossentropy loss)
            pred: model output
            y_a: true y_a
            y_b: true y_a
            lam: lambda (interpolation ratio)

        Returns:
            loss
        """
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    # end of mixup code from: https://arxiv.org/pdf/1710.09412.pdf ==> https://github.com/facebookresearch/mixup-cifar10

    def train_models(self, trainloader, unlabelledloader, unlabelled_sampler, indices_unlabelled, validloader, testloader, modelName, model, optimizer, train_logger, val_logger, num_classes = 10, hard_labeles_for_rotation = {}, init_epoch = 0):
        """
        Method to train the Curriculum Labeling method.
        When no pseudo labels are given: train_base, else: train_pseudo.

        Args:
            trainloader: labeled subset loader
            unlabelledloader: unlabeled subset loader
            unlabelled_sampler: unlabeled subset sampler - useful for pseudo-annotating the data retrieved from unlabelledloader
            indices_unlabelled: indices of the unlabeled set - useful for pseudo-annotating the data retrieved from unlabelledloader
            validloader: validation subset loader
            testloader: test subset loader (when debug is enabled)
            modelName: name given to the model for saving checkpoints 
            model: model instance
            optimizer: predefined optimizer assigned to model
            train_logger: TensorBoard instance logger for the training process
            val_logger: TensorBoard instance logger for the validation process
            num_classes (int, optional): number of classes - bound to dataset and user defined available classes. Defaults to 10.
            hard_labeles_for_rotation (dict, optional): dictionary containing the pseudo annotated samples (sample_index: annotation). Defaults to {}.
            init_epoch (int, optional): initial epoch to start training - could vary when finetuning. Defaults to 0.
        """
        for epoch in range(init_epoch, self.args.epochs + (self.args.lr_rampdown_epochs-self.args.epochs)):
            start_time = time.time()
            if len(hard_labeles_for_rotation) > 0:
                hLabels = np.zeros((len(hard_labeles_for_rotation), self.args.num_classes))
                for i, k in enumerate(hard_labeles_for_rotation):
                    hLabels[i][hard_labeles_for_rotation[k]] = 1
                w = self.get_label_weights(hLabels)
                w = torch.FloatTensor(w/100).cuda()
                self.train_pseudo(unlabelledloader, unlabelled_sampler, indices_unlabelled, hard_labeles_for_rotation, model, optimizer, epoch, self.train_logger, modelName, weights = w, use_zca = self.args.use_zca)
            else:
                w = torch.FloatTensor(np.full(self.args.num_classes, 0.1)).cuda()
                self.train_base(trainloader, model, optimizer, epoch, self.train_logger, use_zca = self.args.use_zca)

            if self.args.swa:
                if epoch > self.args.swa_start and epoch%self.args.swa_freq == 0 :
                    optimizer.swap_swa_sgd()
                    if len(hard_labeles_for_rotation) > 0:
                        optimizer.bn_update(unlabelledloader, model, torch.device("cuda"))
                    else:
                        optimizer.bn_update(trainloader, model, torch.device("cuda"))
                    optimizer.swap_swa_sgd()

            print("--- training " + modelName + " epoch in %s seconds ---" % (time.time() - start_time))

            # evaluate, save best model and log results on console and TensorBoard logger
            self.evaluate_after_train(modelName, validloader, testloader, model, optimizer, epoch)

    def update_args(self, args, model, model_optimizer, update_model=False):
        """
        Useful when the data subsets are updated outside this class.

        Args:
            args (dictionary): dict of parameters set by user or updated by external methods.
        """
        self.args = args
        if update_model:
            self.model = model
            self.model_optimizer = model_optimizer

    def train_iteration(self, iteration=0, image_indices_hard_label={}):
        """
        Train model. Resets the best precision 1 variable to 0 and calls the train_models function.
        Usually, when image_indices_hard_label is empty and iteration is 0, the model is trained using only the labeled subset.

        Args:
            iteration (int, optional): curriculum labeling iteration. Defaults to 0.
            image_indices_hard_label (dict, optional): dictionary of pseudo annotated samples. Defaults to {}.
        """
        self.best_prec1 = 0
        self.train_models(self.args.trainloader, self.args.unlabelledloader, self.args.unlabelled_sampler, self.args.indices_unlabelled, self.args.validloader, self.args.testloader, '{}-Rotation'.format(iteration), \
                    self.model, self.model_optimizer, self.train_logger, self.val_logger, num_classes = self.args.num_classes, hard_labeles_for_rotation = image_indices_hard_label, init_epoch = 0)

    def do_iteration(self, iteration):
        """
        Executes steps 2-5 of the Curriculum Learning algorithm.

        2) Use trained model to get max scores of unlabeled data
        3) Compute threshold (check percentiles_holder parameter) based on max scores -> long tail distribution
        4) Pseudo-label
        5) Train next iteration

        Args:
            iteration (int): curriculum labeling iteration

        Returns:
            image_indices_hard_label: dictionary of pseudo annotated samples.
        """
        # sets mu: percentiles threshold
        percentiles_holder = 100 - (self.args.percentiles_holder) * iteration

        #load best model
        best_features_checkpoint = self.args.exp_dir + '/{}-Rotation.best.ckpt'.format(iteration-1)
        print("=> loading pretrained checkpoint '{}'".format(best_features_checkpoint))
        checkpoint = torch.load(best_features_checkpoint)
        self.best_prec1 = checkpoint['best_prec1']
        self.model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded pretrained checkpoint '{}' (epoch {}, best_prec {})".format(best_features_checkpoint, checkpoint['epoch'], checkpoint['best_prec1']))
        
        # get the results of the unlabeled set evaluated on the trained model
        whole_result = np.asarray(self.get_results_val(self.args.unlabelledloader, self.model))

        # where to hold top scores
        image_indices_hard_label = {}
        max_values_per_image = {}
        max_values = []
        # if algorithm in debug mode, sort and print the most confident values (top scores)
        if self.args.debug:
            whole_result_copy = whole_result.reshape((-1)).copy()
            topScores = np.argpartition(whole_result_copy, -5)[-5:]
            topScores = topScores[np.argsort(whole_result_copy[topScores])]
            prRed('Top Scores: {}'.format(whole_result_copy[topScores]))

        # get top scores
        for i in range (whole_result.shape[0]):
            hardlabel = whole_result[i].argmax()
            max_values.append(whole_result[i][hardlabel])
        # set new threshold based on top scores
        if percentiles_holder < 0:
            percentiles_holder = 0
        threshold = np.percentile(max_values, percentiles_holder)
        prGreen ('Actual Threshold: {} - Percentile: {}'.format(threshold, percentiles_holder))
        # percentiles_holder -= self.args.percentiles_holder

        # check max score against threshold to assign pseudo label
        for i in range (whole_result.shape[0]):
            hardlabel = whole_result[i].argmax()
            # check top percent as threshold
            if whole_result[i][hardlabel] >= threshold:
                image_indices_hard_label[self.args.indices_unlabelled[i]] = hardlabel
            max_values_per_image[i] = [hardlabel, whole_result[i][hardlabel]]

        prGreen ('[{} Rotation] | Total of hard-labeled images: {}'.format((iteration), len(image_indices_hard_label)))

        #add the labeled images
        for i, (_, target) in enumerate(self.args.trainloader):
            for t in range(len(target)):
                image_indices_hard_label[self.args.indices_train[(i*self.args.batch_size)+t]] = target[t].item()

        prGreen ('[{} Rotation] | Total of hard-labeled images + known labeled images: {}'.format(iteration, len(image_indices_hard_label)))
        pickle.dump(image_indices_hard_label, open(self.args.exp_dir + '/{}-Rotation_HardLabels.p'.format(iteration), "wb"))
        pickle.dump(max_values_per_image, open(self.args.exp_dir + '/{}-Rotation_MaxValues.p'.format(iteration), "wb"))

        self.train_logger.scalar_summary('all/images', len(image_indices_hard_label), iteration)

        return image_indices_hard_label

    def train_base(self, trainloader, model, optimizer, epoch, train_logger, use_zca = True, weights = None):
        """
        Train using the labeled subset only.

        Args:
            trainloader: labeled subset data loader
            model: model instance
            optimizer: predefined optimizer assigned to model
            epoch: current epoch
            train_logger: instance to TensorBoard logs
            use_zca (bool, optional): use zca or not (for CIFAR10)
            weights (optional): class weights
        """
        class_criterion = nn.CrossEntropyLoss(weight = weights).cuda()
        meters = AverageMeterSet()
        model.train()
        end = time.time()

        for i, (input, target) in enumerate(trainloader):
            # measure data loading time
            meters.update('data_time', time.time() - end)
            if self.args.dataset == 'cifar10':
                if use_zca:
                    input = apply_zca(input, zca_mean, zca_components)

            if epoch <= self.args.epochs:
                lr = self.adjust_learning_rate(optimizer, epoch, i, len(trainloader))
            meters.update('lr', optimizer.param_groups[0]['lr'])
            input_var = torch.autograd.Variable(input.cuda())
            target_var = torch.autograd.Variable(target.cuda(non_blocking = True))

            # continue with standard training
            self.apply_train_common(model, class_criterion, optimizer, input_var, target_var, i, len(trainloader), epoch, meters, end)
        
        self.train_class_loss_list.append(meters['class_loss'].avg)
        self.train_error_list.append(meters['error1'].avg)
        self.train_lr_list.append(meters['lr'].avg)

        self.train_logger.scalar_summary('0-Rotation/train/loss', meters['class_loss'].avg, epoch)
        self.train_logger.scalar_summary('0-Rotation/train/prec1', meters['top1'].avg, epoch)
        self.train_logger.scalar_summary('0-Rotation/train/prec5', meters['top5'].avg, epoch)
        self.train_logger.scalar_summary('0-Rotation/train/lr', meters['lr'].avg, epoch)

        self.log_img_and_pseudolabels(input[0], target[0], self.train_logger)

    def train_pseudo(self, unlabelledloader, unlabelled_sampler, indices_unlabelled, hardLabeledResults, model, optimizer, epoch, train_logger, modelName, weights = None, use_zca = True):
        """
        Train using pseudo labeled samples along with the labeled subset.

        Args:
            unlabelledloader: unlabel subset data loader
            unlabelled_sampler: unlabel subset data sampler
            indices_unlabelled: indices of unlabeled set
            hardLabeledResults: dictionary containing the pseudo annotated samples (sample_index: annotation)
            model: model instance
            optimizer: predefined optimizer assigned to model
            epoch: current epoch
            train_logger: instance to TensorBoard logs
            use_zca (bool, optional): use zca or not (for CIFAR10)
            weights (optional): class weights
        """
        class_criterion = nn.CrossEntropyLoss(weight = weights).cuda()
        meters = AverageMeterSet()
        model.train()
        end = time.time()

        for i, (input, _) in enumerate(unlabelledloader):
            # measure data loading time
            meters.update('data_time', time.time() - end)
            if i == 0:
                # get indexes to access the pseudo annotations
                unlabOrigIdx = unlabelled_sampler.getOriginalIndices()
            if self.args.dataset == 'cifar10':
                if use_zca:
                    input = apply_zca(input, zca_mean, zca_components)

            if epoch <= self.args.epochs:
                lr = self.adjust_learning_rate(optimizer, epoch, i, len(unlabelledloader))
            meters.update('lr', optimizer.param_groups[0]['lr'])
            input_var = torch.autograd.Variable(input.cuda())

            # now assign the pseudo labels
            newTarget = torch.zeros(len(input), dtype=torch.long)
            for t in range(len(input)):
                # get the image index
                indexInFile = indices_unlabelled[unlabOrigIdx[(i*self.args.batch_size)+t].item()]
                # assign pseudo targets
                fakeLabel = torch.tensor(hardLabeledResults[indexInFile], dtype=torch.long)
                newTarget[t] = fakeLabel
            target_var = torch.autograd.Variable(newTarget.cuda(non_blocking = True))
            # continue with standard training
            self.apply_train_common(model, class_criterion, optimizer, input_var, target_var, i, len(unlabelledloader), epoch, meters, end)

        self.train_class_loss_list.append(meters['class_loss'].avg)
        self.train_error_list.append(meters['error1'].avg)
        self.train_lr_list.append(meters['lr'].avg)

        self.train_logger.scalar_summary('{}/train/loss'.format(modelName), meters['class_loss'].avg, epoch)
        self.train_logger.scalar_summary('{}/train/prec1'.format(modelName), meters['top1'].avg, epoch)
        self.train_logger.scalar_summary('{}/train/prec5'.format(modelName), meters['top5'].avg, epoch)
        self.train_logger.scalar_summary('{}/train/lr'.format(modelName), meters['lr'].avg, epoch)

        self.log_img_and_pseudolabels(input[0], target_var[0], self.train_logger)

    def get_results_val(self, eval_loader, model):
        """
        Get results from the current model evaluated on a subset.

        Args:
            eval_loader: data loader
            model: model instance

        Returns:
            array: contains the results of the evaluated subset (the output of the model after applying a softmax operation)
        """
        setResults = []
        model.eval()

        end = time.time()
        for i, (input, _) in enumerate(eval_loader):

            if self.args.dataset == 'cifar10':
                if self.args.use_zca:
                    input = apply_zca(input, zca_mean, zca_components)

            with torch.no_grad():
                input_var = torch.autograd.Variable(input.cuda())

            output1 = model(input_var)
            softmax1 = F.softmax(output1, dim=1)

            setResults.extend(softmax1.cpu().detach().numpy())

        return setResults

    def evaluate_all_iterations(self):
        iteration = 0
        val_best = 0
        test1_best = 0
        test5_best = 0
        iter_best = 0
        while self.args.percentiles_holder * iteration <= 100:
            print("=> loading pretrained checkpoint '{}'".format(self.args.exp_dir + '/{}-Rotation.best.ckpt'.format(iteration)))
            checkpoint = torch.load(self.args.exp_dir  + '/{}-Rotation.best.ckpt'.format(iteration))
            best_prec1 = checkpoint['best_prec1']
            self.model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded pretrained checkpoint '{}' (epoch {})".format(self.args.exp_dir + '/{}-Rotation.best.ckpt'.format(iteration), checkpoint['epoch']))

            prGreen("Val: Evaluating the model:")
            top1_val, top5_val, loss_val = self.validate(self.args.validloader, self.model, self.args.start_epoch)
            print ('=====================================================')
            prGreen("Test: Evaluating the model:")
            top1_test, top5_test, loss_test = self.validate(self.args.testloader, self.model, self.args.start_epoch)

            if top1_val > val_best:
                val_best = top1_val
                test1_best = top1_test
                test5_best = top5_test
                iter_best = iteration

            iteration += 1

        prGreen('Final top-1 test accuracy: {}, top-5 test accuracy: {} || {}-th iteration'.format(test1_best, test5_best, iter_best))

