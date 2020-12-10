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

class ICT(Train_Base):
    """
    Interpolation Consistency Training (ICT) for Deep Semi-supervised Learning - https://arxiv.org/abs/1903.03825.
    """
    def __init__(self, args):
        """
        Initialize the Curriculum Learning class with all methods and required variables
        This class use the model, optimizer, dataloaders and all the user parameters to train the CL algorithm proposed by Cascante-Bonilla et. al. in Curriculum Learning: (https://arxiv.org/abs/2001.06001)

        Args:
            args (dictionary): all user defined parameters with some pre-initialized objects (e.g., model, optimizer, dataloaders)
        """
        self.best_prec1 = 0
        self.global_step = 0

        ### list error and losses ###
        self.train_class_loss_list = []
        self.train_error_list = []
        self.train_lr_list = []
        self.val_class_loss_list = []
        self.val_error_list = []

        self.train_ema_class_loss_list = []
        self.train_mixup_consistency_loss_list = []
        self.train_mixup_consistency_coeff_list = []
        self.train_ema_error_list = []
        self.val_ema_class_loss_list = []
        self.val_ema_error_list = []

        exp_dir = os.path.join(args['root_dir'], '{}/{}/{}'.format(args['dataset'], args['arch'], args['add_name']))
        prGreen('Results will be saved to this folder: {}'.format(exp_dir))

        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)

        self.args = args
        self.args['exp_dir'] = exp_dir

        # add TF Logger
        self.train_logger = Logger(os.path.join(self.args['exp_dir'], 'TFLogs/train'))
        self.val_logger = Logger(os.path.join(self.args['exp_dir'], 'TFLogs/val'))

    def mixup_data_sup(self, x, y, alpha=1.0):
        '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
        if alpha > 0.:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.
        batch_size = x.size()[0]
        index = np.random.permutation(batch_size)
        mixed_x = lam * x + (1 - lam) * x[index,:]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def mixup_criterion(self, y_a, y_b, lam):
        return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    def mixup_data(self, x, y, alpha=1.0):
        '''Compute the mixup data. Return mixed inputs, mixed target, and lambda'''
        if alpha > 0.:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.
        batch_size = x.size()[0]
        index = np.random.permutation(batch_size)
        x, y = x.data.cpu().numpy(), y.data.cpu().numpy()
        mixed_x = torch.Tensor(lam * x + (1 - lam) * x[index,:])
        mixed_y = torch.Tensor(lam * y + (1 - lam) * y[index,:])

        mixed_x = Variable(mixed_x.cuda())
        mixed_y = Variable(mixed_y.cuda())
        return mixed_x, mixed_y, lam

    def update_ema_variables(self, model, ema_model, alpha, global_step):
        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    def train(self, trainloader, unlabelledloader, model, ema_model, optimizer, epoch, filep):
        class_criterion = nn.CrossEntropyLoss().cuda()
        criterion_u= nn.KLDivLoss(reduction='batchmean').cuda()
        if self.args['consistency_type'] == 'mse':
            consistency_criterion = losses.softmax_mse_loss
        elif self.args['consistency_type'] == 'kl':
            consistency_criterion = losses.softmax_kl_loss
        else:
            assert False, self.args['consistency_type']

        meters = AverageMeterSet()

        # switch to train mode
        model.train()
        ema_model.train()

        end = time.time()
        i = -1
        for (input, target), (u, _) in zip(cycle(trainloader), unlabelledloader):
            # measure data loading time
            i = i+1
            meters.update('data_time', time.time() - end)

            if input.shape[0]!= u.shape[0]:
                bt_size = np.minimum(input.shape[0], u.shape[0])
                input = input[0:bt_size]
                target = target[0:bt_size]
                u = u[0:bt_size]


            if self.args['dataset'] == 'cifar10':
                input = apply_zca(input, self.args['zca_mean'], self.args['zca_components'])
                u = apply_zca(u, self.args['zca_mean'], self.args['zca_components'])
            lr = self.adjust_learning_rate(optimizer, epoch, i, len(unlabelledloader))
            meters.update('lr', optimizer.param_groups[0]['lr'])

            if self.args['mixup_sup_alpha']:
                if self.args['use_cuda']:
                    input , target, u  = input.cuda(), target.cuda(), u.cuda()
                input_var, target_var, u_var = Variable(input), Variable(target), Variable(u)

                mixed_input, target_a, target_b, lam = mixup_data_sup(input, target, self.args['mixup_sup_alpha'])
                #if self.args['use_cuda']:
                #    mixed_input, target_a, target_b  = mixed_input.cuda(), target_a.cuda(), target_b.cuda()
                mixed_input_var, target_a_var, target_b_var = Variable(mixed_input), Variable(target_a), Variable(target_b)
                output_mixed_l = model(mixed_input_var)

                loss_func = mixup_criterion(target_a_var, target_b_var, lam)
                class_loss = loss_func(class_criterion, output_mixed_l)

            else:
                input_var = torch.autograd.Variable(input.cuda())
                with torch.no_grad():
                    u_var = torch.autograd.Variable(u.cuda())
                target_var = torch.autograd.Variable(target.cuda(async=True))
                output = model(input_var)
                class_loss = class_criterion(output, target_var)

            meters.update('class_loss', class_loss.item())

            ### get ema loss. We use the actual samples(not the mixed up samples ) for calculating EMA loss
            minibatch_size = len(target_var)
            ema_logit_unlabeled = ema_model(u_var)
            ema_logit_labeled = ema_model(input_var)
            if self.args['mixup_sup_alpha']:
                class_logit = model(input_var)
            else:
                class_logit = output
            cons_logit = model(u_var)

            ema_logit_unlabeled = Variable(ema_logit_unlabeled.detach().data, requires_grad=False)

            #class_loss = class_criterion(class_logit, target_var) / minibatch_size

            ema_class_loss = class_criterion(ema_logit_labeled, target_var)# / minibatch_size
            meters.update('ema_class_loss', ema_class_loss.item())

            ### get the unsupervised mixup loss###
            if self.args['mixup_consistency']:
                mixedup_x, mixedup_target, lam = mixup_data(u_var, ema_logit_unlabeled, self.args['mixup_usup_alpha'])
                output_mixed_u = model(mixedup_x)

                mixup_consistency_loss = consistency_criterion(output_mixed_u, mixedup_target) / minibatch_size# criterion_u(F.log_softmax(output_mixed_u,1), F.softmax(mixedup_target,1))
                meters.update('mixup_cons_loss', mixup_consistency_loss.item())
                if epoch < self.args['consistency_rampup_starts']:
                    mixup_consistency_weight = 0.0
                else:
                    mixup_consistency_weight = get_current_consistency_weight(self.args['mixup_consistency'], epoch, i, len(unlabelledloader))
                meters.update('mixup_cons_weight', mixup_consistency_weight)
                mixup_consistency_loss = mixup_consistency_weight*mixup_consistency_loss
            else:
                mixup_consistency_loss = 0
                meters.update('mixup_cons_loss', 0)

            loss = class_loss + mixup_consistency_loss
            meters.update('loss', loss.item())

            prec1, prec5 = self.accuracy(class_logit.data, target_var.data, topk=(1, 5))
            meters.update('top1', prec1[0], minibatch_size)
            meters.update('error1', 100. - prec1[0], minibatch_size)
            meters.update('top5', prec5[0], minibatch_size)
            meters.update('error5', 100. - prec5[0], minibatch_size)

            ema_prec1, ema_prec5 = self.accuracy(ema_logit_labeled.data, target_var.data, topk=(1, 5))
            meters.update('ema_top1', ema_prec1[0], minibatch_size)
            meters.update('ema_error1', 100. - ema_prec1[0], minibatch_size)
            meters.update('ema_top5', ema_prec5[0], minibatch_size)
            meters.update('ema_error5', 100. - ema_prec5[0], minibatch_size)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.global_step += 1
            self.update_ema_variables(model, ema_model, self.args['ema_decay'], self.global_step)

            # measure elapsed time
            meters.update('batch_time', time.time() - end)
            end = time.time()

            if i % self.args['print_freq'] == 0:
                print(
                    'Epoch: [{0}][{1}/{2}]\t'
                    'Time {meters[batch_time]:.3f}\t'
                    'Data {meters[data_time]:.3f}\t'
                    'Class {meters[class_loss]:.4f}\t'
                    'Mixup Cons {meters[mixup_cons_loss]:.4f}\t'
                    'Prec@1 {meters[top1]:.3f}\t'
                    'Prec@5 {meters[top5]:.3f}'.format(
                        epoch, i, len(unlabelledloader), meters=meters))
                #print ('lr:',optimizer.param_groups[0]['lr'])
                filep.write(
                    'Epoch: [{0}][{1}/{2}]\t'
                    'Time {meters[batch_time]:.3f}\t'
                    'Data {meters[data_time]:.3f}\t'
                    'Class {meters[class_loss]:.4f}\t'
                    'Mixup Cons {meters[mixup_cons_loss]:.4f}\t'
                    'Prec@1 {meters[top1]:.3f}\t'
                    'Prec@5 {meters[top5]:.3f}'.format(
                        epoch, i, len(unlabelledloader), meters=meters))

        self.train_class_loss_list.append(meters['class_loss'].avg)
        self.train_ema_class_loss_list.append(meters['ema_class_loss'].avg)
        self.train_mixup_consistency_loss_list.append(meters['mixup_cons_loss'].avg)
        self.train_mixup_consistency_coeff_list.append(meters['mixup_cons_weight'].avg)
        self.train_error_list.append(meters['error1'].avg)
        self.train_ema_error_list.append(meters['ema_error1'].avg)
        self.train_lr_list.append(meters['lr'].avg)
