import pytest
import unittest
import numpy as np

from .. import dataloaders

# class ClassTest(unittest.TestCase):

#      @pytest.mark.xfail
#      def test_feature_a(self):
#         self.assertEqual(2, 3)

#      def test_feature_b(self):
#         self.assertTrue(True)


# @pytest.mark.parametrize(
#     "test_input,expected",
#     [("3+5", 8), ("2+4", 6), pytest.param("6*9", 42, marks=pytest.mark.xfail)],
# )
# def test_eval(test_input, expected):
#     assert eval(test_input) == expected

@pytest.mark.parametrize(
    "augPolicy, dataset, data_dir, expected",
    [(0, "cifar10", "/data/cifar10", 10), (1, "cifar100", "data/cifar100", 100), (2, "cifar100", "data/cifar100", 100), pytest.param(-1, "cifar10", "/data/cifar10_v1", 10, marks=pytest.mark.xfail)],
)
def test_datasubset(augPolicy, dataset, data_dir, expected):
	num_classes, train_data, train_data_noT, test_data = dataloaders.load_data_subsets(augPolicy, dataset, data_dir)
	assert num_classes == expected

@pytest.mark.parametrize(
    "augPolicy, dataset, data_dir, expected",
    [(2, "cifar10", "/data/cifar10", 10)],
)
def test_train_dataloaders_10(augPolicy, dataset, data_dir, expected):
	num_classes, train_data, train_data_noT, test_data = dataloaders.load_data_subsets(augPolicy, dataset, data_dir)
	trainloader, \
	validloader, \
	unlabelledloader, \
	train_sampler, \
	unlabelled_sampler, \
	indices_train, \
	indices_unlabelled, \
	trainIndexOrder, \
	unlabeledIndexOrder = dataloaders.get_train_dataloaders(dataset, train_data, train_data_noT, 100, 12, 1000, 1000, 1, [0,1,2,3,4,5,6,7,8,9], [0,1,2,3,4,5,6,7,8,9], ordered=False)
	assert num_classes == expected

@pytest.mark.parametrize(
    "augPolicy, dataset, data_dir, expected",
    [(1, "cifar100", "data/cifar100", 100)], 
)
def test_train_dataloaders_100(augPolicy, dataset, data_dir, expected):
	num_classes, train_data, train_data_noT, test_data = dataloaders.load_data_subsets(augPolicy, dataset, data_dir)
	trainloader, \
	validloader, \
	unlabelledloader, \
	train_sampler, \
	unlabelled_sampler, \
	indices_train, \
	indices_unlabelled, \
	trainIndexOrder, \
	unlabeledIndexOrder = dataloaders.get_train_dataloaders(dataset, train_data, train_data_noT, 100, 12, 1000, 1000, 1, np.arange(100), np.arange(100), ordered=False)
	assert num_classes == expected