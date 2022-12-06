import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable

import numpy as np
import time
import random
from sklearn.metrics import f1_score
from collections import defaultdict

from graphsage.encoders import Encoder
from graphsage.aggregators import MeanAggregator
from graphsage.aggregators import MaxAggregator
from graphsage.aggregators import RandomAggregator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# seed
RANDOM_SEED = 1 # G7
# embedding dimension
EMBED_DIM = 128 # G7
# epsilon
EPSILON = 0.1 # G7
# batches
BATCHES = 1000 #G7
# Epochs
EPOCHS = 1000
BATCHES = 200 # G7
# learning rate
LRATE = .7  # G7

# number of nodes in the cora data
NUM_NODES_CORA = 2708 # G7
# number of features in the cora embeddings
NUM_FEATS_CORA = 1433 # G7
NUM_SAMPLES_CORA_LAYER1 = 5 # G7
NUM_SAMPLES_CORA_LAYER2 = 5 # G7

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


def run_cora():
    feat_data, labels, adj_lists = load_cora()
    features = nn.Embedding(NUM_NODES_CORA, NUM_FEATS_CORA)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
    # features.cuda()

    agg1 = MeanAggregator(features, cuda=True)  # G7
    enc1 = Encoder(features, NUM_FEATS_CORA, EMBED_DIM, adj_lists, agg1, NUM_SAMPLES_CORA_LAYER1, gcn=True, cuda=False)
    agg2 = MaxAggregator(lambda nodes: enc1(nodes).t(), cuda=False) # G7
    enc2 = Encoder(lambda nodes: enc1(nodes).t(), enc1.embed_dim, EMBED_DIM, adj_lists, agg2, NUM_SAMPLES_CORA_LAYER2, base_model=enc1, gcn=True, cuda=False)

    graphsage = SupervisedGraphSage(7, enc2)
    # graphsage.cuda()
    # graphsage.to(device)

    ############# G7
    # times = []

    # for epoch in range(EPOCHS):
    #     rand_indices = np.random.permutation(NUM_NODES_CORA)
    #     test = rand_indices[:1000]
    #     val = rand_indices[1000:1500]
    #     train = list(rand_indices[1500:])


        # rand_indices = np.random.permutation(NUM_NODES_CORA)
        # test = rand_indices[:1000]
        # val = rand_indices[1000:1500]
        # train = list(rand_indices[1500:])

        # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, graphsage.parameters()), lr=0.7)

        # times = []

        # for batch in range(BATCHES):
        #     batch_nodes = train[:256]
        #     random.shuffle(train)
        #     start_time = time.time()

        #     # RANDOM AGGREGATOR #G7
        #     #chooseAggregator(enc1, enc2, features)

        #     optimizer.zero_grad()
        #     loss = graphsage.loss(batch_nodes, Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
        #     loss.backward()
        #     optimizer.step()
        #     end_time = time.time()
        #     times.append(end_time - start_time)

    ############# /G7

    rand_indices = np.random.permutation(NUM_NODES_CORA)
    test = rand_indices[:1000]
    val = rand_indices[1000:1500]
    train = list(rand_indices[1500:])

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, graphsage.parameters()), lr=LRATE)
    times = []

    for batch in range(BATCHES):
        batch_nodes = train[:256]
        random.shuffle(train)
        start_time = time.time()

            # RANDOM AGGREGATOR #G7
            #chooseAggregator(enc1, enc2, features)

        optimizer.zero_grad()
        loss = graphsage.loss(batch_nodes, Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time - start_time)

    print("Epochs: ", EPOCHS)
    start_output = time.time()
    val_output = graphsage.forward(val)
    end_output = time.time()
    print("Validation F1:", f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro"))
    print("Test Time: ", end_output - start_output)
    print("Average batch time:", np.mean(times))


# number of nodes in the pubmed data
NUM_NODES_PUBMED = 19717  # G7
# number of features in pubmed embeddings
NUM_FEATS_PUBMED = 500  # G7
NUM_SAMPLES_PUBMED_LAYER1 = 10  # G7
NUM_SAMPLES_PUBMED_LAYER2 = 25  # G7


def load_pubmed():
    feat_data = np.zeros((NUM_NODES_PUBMED, NUM_FEATS_PUBMED))
    labels = np.empty((NUM_NODES_PUBMED, 1), dtype=np.int64)
    node_map = {}
    with open("pubmed-data/Pubmed-Diabetes.NODE.paper.tab") as fp:
        fp.readline()
        feat_map = {entry.split(":")[1]:i-1 for i,entry in enumerate(fp.readline().split("\t"))}
        for i, line in enumerate(fp):
            info = line.split("\t")
            node_map[info[0]] = i
            labels[i] = int(info[1].split("=")[1])-1
            for word_info in info[2:-1]:
                word_info = word_info.split("=")
                feat_data[i][feat_map[word_info[0]]] = float(word_info[1])
    adj_lists = defaultdict(set)
    with open("pubmed-data/Pubmed-Diabetes.DIRECTED.cites.tab") as fp:
        fp.readline()
        fp.readline()
        for line in fp:
            info = line.strip().split("\t")
            paper1 = node_map[info[1].split(":")[1]]
            paper2 = node_map[info[-1].split(":")[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    return feat_data, labels, adj_lists


def run_pubmed():
    feat_data, labels, adj_lists = load_pubmed()
    features = nn.Embedding(NUM_NODES_PUBMED, NUM_FEATS_PUBMED)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
    # features.cuda()

    agg1 = MeanAggregator(features, cuda=True)
    enc1 = Encoder(features, NUM_FEATS_PUBMED, EMBED_DIM, adj_lists, agg1, NUM_SAMPLES_PUBMED_LAYER1, gcn=True, cuda=False)
    agg2 = MeanAggregator(lambda nodes : enc1(nodes).t(), cuda=False)
    enc2 = Encoder(lambda nodes : enc1(nodes).t(), enc1.embed_dim, EMBED_DIM, adj_lists, agg2, NUM_SAMPLES_PUBMED_LAYER2,
                   base_model=enc1, gcn=True, cuda=False)

    graphsage = SupervisedGraphSage(3, enc2)
    # graphsage.cuda()
    rand_indices = np.random.permutation(NUM_NODES_PUBMED)
    test = rand_indices[:1000]
    val = rand_indices[1000:1500]
    train = list(rand_indices[1500:])

    optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, graphsage.parameters()), lr=LRATE)
    times = []

    for batch in range(200):
        batch_nodes = train[:1024]
        random.shuffle(train)
        start_time = time.time()
        optimizer.zero_grad()
        loss = graphsage.loss(batch_nodes, Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time-start_time)

    val_output = graphsage.forward(val)
    print("Validation F1:", f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro"))
    print("Average batch time:", np.mean(times))


# G7
def chooseAggregator(encoder1, encoder2, features):
    if random.random() < EPSILON:
        encoder1.aggregator = MaxAggregator(features, cuda=False)
        print("Encoder 1: Alt")
    else:
        encoder1.aggregator = MeanAggregator(features, cuda=False)
        print("Encoder 1: Mean")

    if random.random() < EPSILON:
        encoder2.aggregator = MaxAggregator(lambda nodes: encoder1(nodes).t(), cuda=False)
        print("Encoder 2: Alt")
    else:
        encoder2.aggregator = MeanAggregator(lambda nodes: encoder1(nodes).t(), cuda=False)
        print("Encoder 2: Mean")


def init_seeds():
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)


def main():
    init_seeds()
    run_cora()
    #run_pubmed()


if __name__ == "__main__":
    main()
