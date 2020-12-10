import torch
from torch.utils.data.sampler import Sampler

class SubsetSequentialSampler(Sampler):
    """Samples elements from a given list of indices in order, without replacement.
    Has getOriginalIndices: which returns the order of the indexes. 

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices
        self.originalIndices = indices

    def __iter__(self):
        self.originalIndices = range(len(self.indices))
        return (self.indices[i] for i in self.originalIndices)

    def __len__(self):
        return len(self.indices)

    def getOriginalIndices(self):
        return self.originalIndices

class CustomSubsetRandomSampler(Sampler):
    """Samples elements randomly from a given list of indices, without replacement.
    Has getOriginalIndices: which returns the order of the indexes after permuting them.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices
        self.originalIndices = indices

    def __iter__(self):
        self.originalIndices = torch.randperm(len(self.indices))
        return (self.indices[i] for i in self.originalIndices)

    def __len__(self):
        return len(self.indices)

    def getOriginalIndices(self):
        return self.originalIndices