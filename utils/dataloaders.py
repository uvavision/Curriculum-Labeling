import os, errno
import numpy as np
from scipy import linalg
import random
import pickle
from itertools import repeat, cycle

import torch
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms

from .archive import random_augment
from .augmentations import *

from .samplers import *
from .helpers import *


def denorm(x):
    out = (x+1)/2
    return out.clamp(0,1)

def make_dir_if_not_exists(path):
    """Make directory if doesn't already exists"""
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def apply_zca(data, zca_mean, zca_components):
        temp = data.numpy()
        shape = temp.shape
        temp = temp.reshape(-1, shape[1]*shape[2]*shape[3])
        temp = np.dot(temp - zca_mean, zca_components.T)
        temp = temp.reshape(-1, shape[1], shape [2], shape[3])
        data = torch.from_numpy(temp).float()
        return data

class Augmentation(object):
    """
    Apply a subset of random augmentation policies from a set of random transformations
    """
    def __init__(self, policies):
        self.policies = policies

    def __call__(self, img):
        for _ in range(1):
            policy = random.choice(self.policies)
            for name, pr, level in policy:
                if random.random() > pr:
                    continue
                img = apply_augment(img, name, level)
        return img

#cutout transform
class CutoutDefault(object):
    """
    Apply cutout transformation.
    Code taken from: https://github.com/quark0/darts/blob/master/cnn/utils.py
    """
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img

def load_data_subsets(data_aug, dataset, data_target_dir):
    if dataset == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif dataset == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
    elif dataset == 'svhn':
        mean = [x / 255 for x in [127.5, 127.5, 127.5]]
        std = [x / 255 for x in [127.5, 127.5, 127.5]]
    elif dataset == 'imagenet':
        mean = [x / 255 for x in [123.675, 116.28, 103.53]]
        std = [x / 255 for x in [58.395, 57.12, 57.375]]
    elif dataset == 'cub2011':
        mean = [x / 255 for x in [0.485, 0.456, 0.406]]
        std = [x / 255 for x in [0.229, 0.224, 0.225]]
    elif dataset == 'mnist':
        pass
    else:
        assert False, "Unknow dataset : {}".format(dataset)

    if data_aug==1 or data_aug == 2:
        if dataset == 'cifar10' or dataset == 'cifar100':
            train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean, std)])

            if data_aug==2:
                prRed ('heavy random data augmentation will be applied')
                train_transform.transforms.insert(0, Augmentation(random_augment())) #adding random_augmentations
                train_transform.transforms.append(CutoutDefault(16))
            test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

        if dataset == 'svhn':
            train_transform = transforms.Compose([ transforms.RandomCrop(32, padding=2),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean, std)])
            
            if data_aug==2:
                prRed ('heavy random data augmentation will be applied')
                train_transform.transforms.insert(0, Augmentation(random_augment())) #adding random augmentations
                train_transform.transforms.append(CutoutDefault(16)) #adding cutout
            test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

        if dataset == 'imagenet':
            if data_aug==1:
                train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean, std)])
            
            elif data_aug==2:
                prRed ('heavy random data augmentation will be applied')
                train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                                    transforms.ColorJitter(
                                                        brightness=0.4,
                                                        contrast=0.4,
                                                        saturation=0.4,
                                                        hue=0.2,
                                                    ),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean, std)])
                train_transform.transforms.insert(0, Augmentation(random_augment())) #adding random augmentations
                train_transform.transforms.append(CutoutDefault(20)) #adding cutout

            test_transform = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean, std)])
    else:
        print ('no data aug')
        if dataset == 'mnist':
            hw_size = 28
            train_transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                           ])
            test_transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                           ])

        else:
            train_transform = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize(mean, std)])
            test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    print ('Directory: {}'.format(data_target_dir))

    if dataset == 'cifar10':
        train_data = datasets.CIFAR10(data_target_dir, train=True, transform=train_transform, download=True)
        train_data_noT = datasets.CIFAR10(data_target_dir, train=True, transform=test_transform, download=True)
        test_data = datasets.CIFAR10(data_target_dir, train=False, transform=test_transform, download=True)
        num_classes = 10
    elif dataset == 'cifar100':
        train_data = datasets.CIFAR100(data_target_dir, train=True, transform=train_transform, download=True)
        train_data_noT = datasets.CIFAR100(data_target_dir, train=True, transform=train_transform, download=True)
        test_data = datasets.CIFAR100(data_target_dir, train=False, transform=test_transform, download=True)
        num_classes = 100
    elif dataset == 'svhn':
        train_data = datasets.SVHN(data_target_dir, split='train', transform=train_transform, download=True)
        train_data_noT = datasets.SVHN(data_target_dir, split='train', transform=test_transform, download=True)
        test_data = datasets.SVHN(data_target_dir, split='test', transform=test_transform, download=True)
        num_classes = 10
    elif dataset == 'imagenet':
        train_data = torchvision.datasets.ImageNet(data_target_dir, split='train', transform=train_transform) #, download=True)
        train_data_noT = torchvision.datasets.ImageNet(data_target_dir, split='train', transform=test_transform) #, download=True)
        test_data = torchvision.datasets.ImageNet(data_target_dir, split='val', transform=test_transform) #, download=True)
        num_classes = 1000
    else:
        assert False, 'Do not support dataset : {}'.format(dataset)

    # n_labels = num_classes
    return [num_classes, train_data, train_data_noT, test_data]

def get_sampler(labels, set_labeled_classes, set_unlabeled_classes, n=None, n_valid= None, ordered = False, seed = 0, indices_for_rotation = []):
    # labels available in the dataset (10 for CIFAR10 (6 if only animals) and SVHN, 1000 for Imagenet)
    # n = number of labels per class for training
    # n_val = number of lables per class for validation

    all_train_data = {}
    for i, l in enumerate(labels):
        if l not in list(all_train_data.keys()):
            all_train_data[l] = [i]
        else:
            all_train_data[l].append(i)

    #get pseudo-random distribution of data
    for i in range (len(all_train_data)):
        random.seed(seed)
        randomIndexes = random.sample(range(0, len(all_train_data[i])), len(all_train_data[i]))
        trainData = np.asarray(all_train_data[i])
        all_train_data[i] = trainData[randomIndexes]

    indices_train = []
    indices_unlabelled = []
    indices_valid = []

    for i in range (len(all_train_data)):
        if i in set_labeled_classes:
            indices_train.extend(all_train_data[i][:n])
            indices_valid.extend(all_train_data[i][n:n+n_valid])
        if i in set_unlabeled_classes:
            if i in set_labeled_classes:
                indices_unlabelled.extend(all_train_data[i][n+n_valid:])
            else:
                indices_unlabelled.extend(all_train_data[i][:])
    # else:
    #     for i in range (all_train_data)):
    #         indices_train.extend(all_train_data[i][:n])
    #         indices_valid.extend(all_train_data[i][n:n+n_valid])
    #         indices_unlabelled.extend(all_train_data[i][n+n_valid:])

    #         # indices_valid.extend(all_train_data[i][:n])
    #         # indices_unlabelled.extend(all_train_data[i][n:])

    indices_train = np.asarray(indices_train)
    indices_unlabelled = np.asarray(indices_unlabelled)
    indices_valid = np.asarray(indices_valid)

    #reset my new unlabeled images:
    if len(indices_for_rotation) > 0:
        indices_unlabelled = np.asarray(indices_for_rotation)

    print (indices_train.shape)
    print (indices_unlabelled.shape)
    print (indices_valid.shape)
    print (indices_train[:10])
    print (indices_unlabelled[:10])
    print (indices_valid[:10])

    if ordered:
        print ('-- ordered to get features and scores')
        sampler_train = SubsetSequentialSampler(indices_train)
        train_index_order = sampler_train.getOriginalIndices()
        sampler_valid = SubsetSequentialSampler(indices_valid)
        sampler_unlabelled = SubsetSequentialSampler(indices_unlabelled)
        unlabeled_index_order = sampler_unlabelled.getOriginalIndices()
    else:
        print ('-- not ordered to train')
        sampler_train = CustomSubsetRandomSampler(indices_train)
        train_index_order = sampler_train.getOriginalIndices()
        sampler_valid = CustomSubsetRandomSampler(indices_valid)
        sampler_unlabelled = CustomSubsetRandomSampler(indices_unlabelled)
        unlabeled_index_order = sampler_unlabelled.getOriginalIndices()

    return sampler_train, sampler_valid, sampler_unlabelled, indices_train, indices_unlabelled, train_index_order, unlabeled_index_order

# Returns dataloaders with samplers and index order (for pseudo-labeling)
def get_train_dataloaders(dataset, train_data, train_data_noT, batch_size, workers, labels_per_class, valid_labels_per_class, seed, set_labeled_classes, set_unlabeled_classes, ordered=False, indices_for_rotation = []):

    if dataset == 'svhn':
        train_sampler, valid_sampler, unlabelled_sampler, indices_train, indices_unlabelled, train_index_order, unlabeled_index_order = get_sampler(train_data.labels, set_labeled_classes, set_unlabeled_classes, labels_per_class, valid_labels_per_class, ordered=ordered, seed = seed, indices_for_rotation=indices_for_rotation)
    elif dataset == 'imagenet':
        imgs = train_data.imgs
        allimgs = np.asarray(imgs)
        allsamples = allimgs[:, 1]
        labels = list(map(int, allsamples))
        train_sampler, valid_sampler, unlabelled_sampler, indices_train, indices_unlabelled, train_index_order, unlabeled_index_order = get_sampler(labels, set_labeled_classes, set_unlabeled_classes, labels_per_class, valid_labels_per_class, ordered=ordered, seed = seed, indices_for_rotation=indices_for_rotation)
    elif dataset == 'cub2011':
        train_sampler, valid_sampler, unlabelled_sampler, indices_train, indices_unlabelled, train_index_order, unlabeled_index_order = get_sampler(train_data.data['target'].tolist(), set_labeled_classes, set_unlabeled_classes, labels_per_class, valid_labels_per_class, ordered=ordered, seed = seed, indices_for_rotation=indices_for_rotation)
    else: #cifar10
        train_sampler, valid_sampler, unlabelled_sampler, indices_train, indices_unlabelled, train_index_order, unlabeled_index_order = get_sampler(train_data.targets, set_labeled_classes, set_unlabeled_classes, labels_per_class, valid_labels_per_class, ordered=ordered, seed = seed, indices_for_rotation=indices_for_rotation)

    if ordered: #use moderate transforms to get scores
        labelled = torch.utils.data.DataLoader(train_data_noT, batch_size=batch_size, sampler = train_sampler,  num_workers=workers, pin_memory=True)
        unlabelled = torch.utils.data.DataLoader(train_data_noT, batch_size=batch_size, sampler = unlabelled_sampler,  num_workers=workers, pin_memory=True)
        validation = torch.utils.data.DataLoader(train_data_noT, batch_size=batch_size, sampler = valid_sampler,  num_workers=workers, pin_memory=True)
    else: #use augmentation policy | validation split keeps the moderate transforms
        labelled = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler = train_sampler,  num_workers=workers, pin_memory=True)
        unlabelled = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler = unlabelled_sampler,  num_workers=workers, pin_memory=True)
        validation = torch.utils.data.DataLoader(train_data_noT, batch_size=batch_size, sampler = valid_sampler,  num_workers=workers, pin_memory=True)

    return [labelled, validation, unlabelled, train_sampler, unlabelled_sampler, indices_train, indices_unlabelled, train_index_order, unlabeled_index_order]

def get_test_dataloader(test_data, batch_size, workers):
    test = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    return test
