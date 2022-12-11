# Note: blocks of code marked with the following were added by the members of group 7
####### G7
####### /G7


import datetime
import numpy as np
import sys
import random
import time
import torch
import torch.nn as nn


from collections import defaultdict
from graphsage.encoders import Encoder
from graphsage.aggregators import MeanAggregator
from graphsage.aggregators import MaxAggregator
from sklearn.metrics import f1_score
from torch.nn import init
from torch.autograd import Variable


######## G7
# seed
RANDOM_SEED = 1 
# embedding dimension
EMBED_DIM = 128 
# epsilon
EPSILON = 0.1 
# Epochs
EPOCHS = 10
# Batches
BATCHES = 100 
# learning rate
LRATE = .7  

# number of nodes in the cora data
NUM_NODES_CORA = 2708 
# number of features in the cora embeddings
NUM_FEATS_CORA = 1433 
NUM_SAMPLES_CORA_LAYER1 = 5 
NUM_SAMPLES_CORA_LAYER2 = 5 

######## /G7

"""
Simple supervised GraphSAGE model as well as examples running the model
on the Cora and Pubmed datasets.
"""


class SupervisedGraphSage(nn.Module):
    def __init__(self, num_classes, enc):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc
        self.xent = nn.CrossEntropyLoss()
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))
        init.xavier_uniform_(self.weight)


    def forward(self, nodes):
        embeds = self.enc(nodes)
        scores = self.weight.mm(embeds)
        return scores.t()


    def loss(self, nodes, labels):
        scores = self.forward(nodes)
        return self.xent(scores, labels.squeeze())


def load_cora():
    feat_data = np.zeros((NUM_NODES_CORA, NUM_FEATS_CORA))
    labels = np.empty((NUM_NODES_CORA, 1), dtype=np.int64)
    node_map = {}
    label_map = {}
    with open("cora/cora.content") as fp:
        for i, line in enumerate(fp):
            info = line.strip().split()
            feat_data[i, :] = list(map(float, info[1:-1]))
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]

    adj_lists = defaultdict(set)
    with open("cora/cora.cites") as fp:
        for i, line in enumerate(fp):
            info = line.strip().split()
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    return feat_data, labels, adj_lists


def run_cora(aggr1, aggr2):
    feat_data, labels, adj_lists = load_cora()
    features = nn.Embedding(NUM_NODES_CORA, NUM_FEATS_CORA)  # embeddings are randomly initialized
    # weights are initialized as the 0/1 word vectors
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)

    agg1 = MaxAggregator(features, cuda=False) if aggr1 == "max" else MeanAggregator(features, cuda=False) # G7
    enc1 = Encoder(features, NUM_FEATS_CORA, EMBED_DIM, adj_lists, agg1, NUM_SAMPLES_CORA_LAYER1, gcn=True, cuda=False)
    agg2 = MaxAggregator(lambda nodes: enc1(nodes).t(), cuda=False) if aggr1 == "max" else MeanAggregator(lambda nodes: enc1(nodes).t(), cuda=False) # G7
    enc2 = Encoder(lambda nodes: enc1(nodes).t(), enc1.embed_dim, EMBED_DIM, adj_lists, agg2, NUM_SAMPLES_CORA_LAYER2, base_model=enc1, gcn=True, cuda=False)

    graphsage = SupervisedGraphSage(7, enc2)

    ############# G7
    times = []

    rand_indices = np.random.permutation(NUM_NODES_CORA)
    test = rand_indices[0:1000]
    train = list(rand_indices[2000:])

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, graphsage.parameters()), lr=0.7)

    for epoch in range(EPOCHS):
        for batch in range(BATCHES):
            batch_nodes = train[:256]
            random.shuffle(train)
            start_time = time.time()

            optimizer.zero_grad()
            loss = graphsage.loss(batch_nodes, Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
            loss.backward()
            optimizer.step()
            end_time = time.time()
            times.append(end_time - start_time)

    ############# /G7

    test_output = graphsage.forward(test)
    return (f1_score(labels[test], test_output.data.numpy().argmax(axis=1), average="micro"), np.mean(times))


########## G7
def format_output(f1_scores, average_batch_times):
    print("Epochs: ", EPOCHS)
    print("                                  Aggregator Combinations")
    print("                         ========================================")
    print("Layer 1:                 Max     |  Mean   |   Mean  |    Max")
    print("Layer 2:                 Max     |  Max    |   Mean  |    Mean")
    print("                         ----------------------------------------")
    print("F1 Scores:               {0:.3f}   |  {1:.3f}  |  {2:.3f}  |  {3:.3f}".format(f1_scores[0], f1_scores[1], f1_scores[2], f1_scores[3]))  
    print("Average Batch Times:     {0:.3f}   |  {1:.3f}  |  {2:.3f}  |  {3:.3f}".format(average_batch_times[0], average_batch_times[1], average_batch_times[2], average_batch_times[3]))

def init_seeds():
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)


def main():
    if len(sys.argv) <= 1 or len(sys.argv) > 3: # Invalid input
        print("Usage: python -m graphsage.model aggr1 aggr2 OR python -m graphsage.model all")
        exit(1)
    
    init_seeds()
    if len(sys.argv) == 2: # Run every aggregator combination
        f1_scores = []
        average_batch_times = []

        run1 = run_cora("max", "max")
        run2 = run_cora("mean", "max")
        run3 = run_cora("mean", "mean")
        run4 = run_cora("max", "mean")

        f1_scores.append(run1[0])
        average_batch_times.append(run1[1])

        f1_scores.append(run2[0])
        average_batch_times.append(run2[1])

        f1_scores.append(run3[0])
        average_batch_times.append(run3[1])

        f1_scores.append(run4[0])
        average_batch_times.append(run4[1])

        format_output(f1_scores, average_batch_times)
    else: # Run a specific aggregator combination
        (f1_score, average_batch_time) = run_cora(sys.argv[1], sys.argv[2])
        print("Epochs: ", EPOCHS)
        print("Validation F1:", f1_score)
        print("Average batch time:", average_batch_time)

    exit(0)

if __name__ == "__main__":
    main()

######## /G7