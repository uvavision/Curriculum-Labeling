import os
import time
import numpy as np
import pickle
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.scheduler_ramps import *
from utils.helpers import *

class Train_Base():
    """
    Class with base train methods.
    """

    def __init__(self, args):
        """
        Initializes the class. Assign all parameters including the model, dataloaders, samplers and extra variables for each method.

        Args:
            args (dictionary): all user defined parameters with some pre-initialized objects (e.g., model, optimizer, dataloaders)
        """
        self.args = args

    def apply_train_common(self, model, class_criterion, optimizer, input_var, target_var, loader_index, len_trainloader, epoch, meters, end):
        """
        Common train set of operations shared between all training methods (with labeled and/or pseudo-annotated data)

        Args:
            model: model instance
            class_criterion: categorical crossentropy loss
            optimizer: predefined optimizer assigned to model
            input_var: image samples
            target_var: corresponding targets
            loader_index: index of the data loader
            len_trainloader: length of the data loader
            epoch: current epoch
            meters: AverageMeterSet instance
            end: value passed to measure time
        """
        minibatch_size = len(target_var)

        # mixup added
        if self.args.mixup:
            inputs, targets_a, targets_b, lam = self.mixup_data(input_var, target_var, self.args.alpha)
            input_var, target_a, target_b = map(torch.autograd.Variable, (inputs, targets_a, targets_b))

        model_out = model(input_var)

        if self.args.mixup:
            loss = self.mixup_criterion(class_criterion, model_out, targets_a, targets_b, lam)
        else:
            loss = class_criterion(model_out, target_var)

        meters.update('class_loss', loss.item())

        assert not (np.isnan(loss.item())), 'Loss explosion: {}'.format(loss.data[0])
        meters.update('loss', loss.item())

        prec1, prec5 = self.accuracy(model_out.data, target_var.data, topk=(1, 5))
        meters.update('top1', prec1[0], minibatch_size)
        meters.update('error1', 100. - prec1[0], minibatch_size)
        meters.update('top5', prec5[0], minibatch_size)
        meters.update('error5', 100. - prec5[0], minibatch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if self.args.swa:
            if epoch > self.args.swa_start and epoch%self.args.swa_freq == 0 :
                optimizer.update_swa()

        # measure elapsed time
        meters.update('batch_time', time.time() - end)
        end = time.time()

        if loader_index % self.args.print_freq == 0:

            print(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time {meters[batch_time]:.3f}\t'
                'Data {meters[data_time]:.3f}\t'
                'Loss {meters[loss]:.4f}\t'
                'Prec@1 {meters[top1]:.3f}\t'
                'Prec@5 {meters[top5]:.3f}'.format(
                    epoch, loader_index, len_trainloader, meters=meters))

    def log_img_and_pseudolabels(self, input_img, pseudolabel, logger):
        """
        Add image and label or pseudo-label to TensorBoard log for debugging purposes.

        Args:
            input_img: image
            pseudolabel: label
            logger: log reference
        """
        anns = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        img = input_img.clone()
        img = img.squeeze() # get rid of batch dim.
        if self.args.dataset == 'cifar10':
            # un-normalize pixel values.
            mean = [x / 255 for x in [125.3, 123.0, 113.9]]
            std = [x / 255 for x in [63.0, 62.1, 66.7]]
            for t, m, s in zip(img, mean, std):
                t.mul_(s).add_(m)
            _label = anns[pseudolabel.cpu().detach().numpy()]
            _orig_image = img.transpose(0,2).transpose(0,1).cpu().detach().numpy()
            logger.image_summary(_label, [_orig_image], step=0)
            
            import matplotlib.pyplot as plt
            plt.imshow(_orig_image)
            plt.title(_label)
            plt.savefig('test.png')
            print ('test.png saved')
            # breakpoint()

        elif self.args.dataset == 'imagenet':
            for t, m, s in zip(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]):
                t.mul_(s).add_(m)
            logger.image_summary(pseudolabel.cpu().detach().numpy(), [img.transpose(0,2).transpose(0,1).cpu().detach().numpy()], step=0)        

    def get_label_weights(self, labels):
        """
        Compute class weights. Useful when training unbalanced classes.

        Args:
            labels: labels of all samples

        Returns:
            weights: class weights
        """
        return (labels == 0).sum(axis = 0) / labels.sum(axis = 0)

    def adjust_learning_rate(self, optimizer, epoch, step_in_epoch, total_steps_in_epoch):
        """
        Adjust learning rate based on lr_rampup and lr_rampdown_epochs parameters pre-defined in self.args.

        Args:
            optimizer: predefined optimizer assigned to model
            epoch: current training epoch
            step_in_epoch: current step in epoch
            total_steps_in_epoch: total steps in epoch

        Returns:
            float: new learning rate
        """
        lr = self.args.lr
        epoch = epoch + step_in_epoch / total_steps_in_epoch

        # LR warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677
        lr = linear_rampup(epoch, self.args.lr_rampup) * (self.args.lr - self.args.initial_lr) + self.args.initial_lr

        # Cosine LR rampdown from https://arxiv.org/abs/1608.03983 (but one cycle only)
        if self.args.lr_rampdown_epochs:
            assert self.args.lr_rampdown_epochs >= self.args.epochs
            lr *= cosine_rampdown(epoch, self.args.lr_rampdown_epochs)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        return lr

    def validate(self, eval_loader, model, epoch, testing = False, use_zca = True, k=5):
        """
        Returns current model top-1 and top-k accuracy evaluated on a data loader

        Args:
            eval_loader: data loader - usually validation loader
            model: model instance
            epoch: current epoch
            testing (bool, optional): Defaults to False.
            use_zca (bool, optional): use zca preprocessing (on CIFAR10). Defaults to True.
            k (int, optional): refers to k in top-k accuracy. Defaults to 5.

        Returns:
            float, float, float: returns top-1 acc, top-k acc and loss value
        """
        class_criterion = nn.CrossEntropyLoss().cuda()
        meters = AverageMeterSet()

        # switch to evaluate mode
        model.eval()

        end = time.time()
        for i, (input, target) in enumerate(eval_loader):
            meters.update('data_time', time.time() - end)

            if self.args.dataset == 'cifar10':
                if self.args.use_zca:
                    input = apply_zca(input, zca_mean, zca_components)

            with torch.no_grad():
                input_var = torch.autograd.Variable(input.cuda())
            with torch.no_grad():
                target_var = torch.autograd.Variable(target.cuda(non_blocking = True))

            minibatch_size = len(target_var)

            # compute output
            output1 = model(input_var)
            softmax1 = F.softmax(output1, dim=1)
            class_loss = class_criterion(output1, target_var)

            # measure accuracy and record loss
            prec1, prec5 = self.accuracy(output1.data, target_var.data, topk=(1, k))
            meters.update('class_loss', class_loss.item(), minibatch_size)
            meters.update('top1', prec1[0], minibatch_size)
            meters.update('error1', 100.0 - prec1[0], minibatch_size)
            meters.update('top5', prec5[0], minibatch_size)
            meters.update('error5', 100.0 - prec5[0], minibatch_size)

            # measure elapsed time
            meters.update('batch_time', time.time() - end)
            end = time.time()

        print(' * Prec@1 {top1.avg:.3f}\tPrec@{k} {top5.avg:.3f}'
            .format(top1=meters['top1'], k=k, top5=meters['top5']))

        if testing == False:
            self.val_class_loss_list.append(meters['class_loss'].avg)
            self.val_error_list.append(meters['error1'].avg)

        return meters['top1'].avg, meters['top5'].avg, meters['class_loss'].avg

    def evaluate_after_train(self, modelName, validloader, testloader, model, optimizer, epoch):
        """
        Evaluate and save weights if current validation accuracy is better than previous epoch.
        Log results on console and TensorBoard logger.
        """
        start_time = time.time()
        print("Evaluating the {} model on validation set:".format(modelName))
        prec1, prec5, loss = self.validate(validloader, model, epoch + 1)

        self.val_logger.scalar_summary(modelName + '/val/loss', loss, epoch)
        self.val_logger.scalar_summary(modelName + '/val/prec1', prec1, epoch)
        self.val_logger.scalar_summary(modelName + '/val/prec5', prec5, epoch)

        if self.args.debug:
            prec1_test, prec5_test, loss_test = self.validate(testloader, model, epoch + 1)
            self.val_logger.scalar_summary(modelName + '/test/loss', loss_test, epoch)
            self.val_logger.scalar_summary(modelName + '/test/prec1', prec1_test, epoch)
            self.val_logger.scalar_summary(modelName + '/test/prec5', prec5_test, epoch)

        is_best = prec1 >= self.best_prec1
        self.best_prec1 = max(prec1, self.best_prec1)
        if self.args.checkpoint_epochs and (epoch + 1) % self.args.checkpoint_epochs == 0:
            self.save_checkpoint({
                'epoch': epoch + 1,
                'arch': self.args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': self.best_prec1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, self.args.exp_dir, epoch + 1, modelName)

        if is_best:
            #CHECK PREVIOUS BEST MODEL#
            try:
                best_features_checkpoint = self.args.exp_dir + '/{}.best.ckpt'.format(modelName)
                checkpoint = torch.load(best_features_checkpoint)
                latest_best_prec1 = checkpoint['best_prec1']
                print("=> loaded pretrained checkpoint '{}' (epoch {}, best_prec {})".format(best_features_checkpoint, checkpoint['epoch'], checkpoint['best_prec1']))
                if self.best_prec1 >= latest_best_prec1:
                    prPurple("Current acc: {} is better than previous: {}, save it".format(self.best_prec1, latest_best_prec1))
                    self.save_best_checkpoint({
                        'epoch': epoch + 1,
                        'arch': self.args.arch,
                        'state_dict': model.state_dict(),
                        'best_prec1': self.best_prec1,
                        'optimizer' : optimizer.state_dict(),
                    }, is_best, self.args.exp_dir, epoch + 1, modelName)
            except:
                prPurple("New best checkpoint: {} for model {}".format(self.best_prec1, modelName))
                self.save_best_checkpoint({
                    'epoch': epoch + 1,
                    'arch': self.args.arch,
                    'state_dict': model.state_dict(),
                    'best_prec1': self.best_prec1,
                    'optimizer' : optimizer.state_dict(),
                }, is_best, self.args.exp_dir, epoch + 1, modelName)

        train_log = OrderedDict()
        train_log['train_class_loss_list'] = self.train_class_loss_list
        train_log['train_error_list'] = self.train_error_list
        train_log['train_lr_list'] = self.train_lr_list
        train_log['val_class_loss_list'] = self.val_class_loss_list
        train_log['val_error_list'] = self.val_error_list

        pickle.dump(train_log, open(os.path.join(self.args.exp_dir, modelName + '.pkl'), 'wb'))

    def save_checkpoint(self, state, is_best, dirpath, epoch, modelName):
        """
        Save model weights - checkpoint model

        Args:
            state: current state
            is_best: if the model is best after validation
            dirpath: path to save weights
            epoch: current epoch
            modelName: name to save
        """
        filename = '{}.checkpoint.{}.ckpt'.format(modelName, epoch)
        checkpoint_path = os.path.join(dirpath, filename)
        torch.save(state, checkpoint_path)

    def save_best_checkpoint(self, state, is_best, dirpath, epoch, modelName):
        """
        Save model weights - checkpoint current best model state

        Args:
            state: current state
            is_best: if the model is best after validation
            dirpath: path to save weights
            epoch: current epoch
            modelName: name to save
        """
        best_path = os.path.join(dirpath, '{}.best.ckpt'.format(modelName))
        torch.save(state, best_path)

    def accuracy(self, output, target, topk=(1,)):
        """
        Computes the precision-k for the specified values of k
        """
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

    #############################################################################################################
    # mixup code from: https://arxiv.org/pdf/1710.09412.pdf ==> https://github.com/facebookresearch/mixup-cifar10
    def mixup_data(self, x, y, alpha=1.0, use_cuda=True):
        '''Returns mixed inputs, pairs of targets, and lambda'''
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
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    # end of mixup code from: https://arxiv.org/pdf/1710.09412.pdf
    #############################################################################################################      