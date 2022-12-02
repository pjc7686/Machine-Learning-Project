import torch
import torch.nn as nn
from torch.autograd import Variable

import random

"""
Set of modules for aggregating embeddings of neighbors.
"""


class MaxAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings
    """
    def __init__(self, features, cuda=False, gcn=False):
        """
        Initializes the aggregator for a specific graph.

        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
        """

        super(MaxAggregator, self).__init__()
        self.features = features
        self.cuda = cuda
        self.gcn = gcn

    def forward(self, nodes, to_neighs, num_sample=10):
        """
        nodes --- list of nodes in a batch
        to_neighs --- list of sets, each set is the set of neighbors for node in batch
        num_sample --- number of neighbors to sample. No sampling if None.
        """
        # Local pointers to functions (speed hack)
        _set = set
        _list = list
        if num_sample is not None:
            _sample = random.sample
            samp_neighs = [_list(_sample(to_neigh, num_sample,)) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
        else:
            samp_neighs = to_neighs

        if self.gcn:
            samp_neighs = [samp_neigh + {[nodes[i]]} for i, samp_neigh in enumerate(samp_neighs)]

        num_features = len(self.features(0))
        embed_matrix = Variable(torch.zeros(len(samp_neighs), num_features))
        for i, nbrs in enumerate(samp_neighs):
            nbrs = list(nbrs)
            samp_neighs[i] = nbrs[random.randrange(len(nbrs))]
            embeds = self.features(torch.LongTensor(nbrs))
            max_feature = torch.max(embeds)
            embed_matrix[i] = max_feature

        to_feats = embed_matrix # we need to_feats to be 635 (for each input node) by 1433 (for each feature)
        return to_feats


class RandomAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings
    """
    def __init__(self, features, cuda=False, gcn=False):
        """
        Initializes the aggregator for a specific graph.

        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
        """

        super(RandomAggregator, self).__init__()
        self.features = features
        self.cuda = cuda
        self.gcn = gcn

    def forward(self, nodes, to_neighs, num_sample=10):
        """
        nodes --- list of nodes in a batch
        to_neighs --- list of sets, each set is the set of neighbors for node in batch
        num_sample --- number of neighbors to sample. No sampling if None.
        """
        # Local pointers to functions (speed hack)
        _set = set
        if num_sample is not None:
            _sample = random.sample
            samp_neighs = [_set(_sample(to_neigh, num_sample,)) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
        else:
            samp_neighs = to_neighs

        if self.gcn:
            samp_neighs = [samp_neigh + {[nodes[i]]} for i, samp_neigh in enumerate(samp_neighs)]


        for i, samp in enumerate(samp_neighs):
            samp = list(samp)
            samp_neighs[i] = samp[random.randrange(len(samp))]
        embed_matrix = self.features(torch.LongTensor(samp_neighs))

        to_feats = embed_matrix # we need to_feats to be 635 (for each input node) by 1433 (for each feature)
        return to_feats


class MeanAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings
    """
    def __init__(self, features, cuda=False, gcn=False): 
        """
        Initializes the aggregator for a specific graph.

        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
        """

        super(MeanAggregator, self).__init__()
        self.features = features
        self.cuda = cuda
        self.gcn = gcn
        
    def forward(self, nodes, to_neighs, num_sample=10):
        """
        nodes --- list of nodes in a batch
        to_neighs --- list of sets, each set is the set of neighbors for node in batch
        num_sample --- number of neighbors to sample. No sampling if None.
        """
        # Local pointers to functions (speed hack)
        _set = set

        # If the hyperparameter k is set, sample k neighbors from the adjacency list, otherwise use all neighbors
        if num_sample is not None:
            _sample = random.sample
            samp_neighs = [_set(_sample(to_neigh, num_sample,)) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
        else:
            samp_neighs = to_neighs

        if self.gcn:
            samp_neighs = [samp_neigh + {[nodes[i]]} for i, samp_neigh in enumerate(samp_neighs)]

        unique_nodes_list = list(set.union(*samp_neighs)) # returns all distinct nodes from sampled neighbors
        unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)} # turn this list into a dictionary
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes))) # create a tensor mask of 0s and 1s
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
        row_indices = [i for i in range(len(samp_neighs)) for _ in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1 #creates adj list of sorts?

        if self.cuda:
            mask = mask.cuda()

        num_neigh = mask.sum(1, keepdim=True) # sum neighbors in order to find number of neighbors
        # num_neigh = tensor of size 256 by 1
        mask = mask.div(num_neigh) # divide each value by number of neighbors (for avg calculation I believe)

        if self.cuda:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())
        else:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list)) # 1309 (num unique nodes) x 1433 (num feats)

        to_feats = mask.mm(embed_matrix) # use matrix multiply to sum features. Values are already divided by mask

        return to_feats
