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
from graphsage.classify import classify
from graphsage.gumbel import gumbel_softmax

"""
Simple supervised GraphSAGE model as well as examples running the model
on the Cora and Pubmed datasets.
"""
import torch 
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn 
from torch.nn import init

import sys 

NUM_SAMPLED = int(sys.argv[1])

def fixed_unigram_candidate_sampler(num_sampled, unique, range_max, distortion, unigrams):
    weights = unigrams**distortion
    prob = weights/weights.sum()
    sampled = np.random.choice(range_max, num_sampled, p=prob, replace=~unique)
    return sampled

def sample_negative(anchors, neg_adj_list):
    return [random.choice(neg_adj_list[anchor]) for anchor in anchors]
        
def sigmoid_cross_entropy_with_logits(labels, logits):
    sig_aff = torch.sigmoid(logits)
    loss = labels * -torch.log(sig_aff+1e-10) + (1 - labels) * -torch.log(1 - sig_aff+1e-10)
    return loss

def node2vec(outputs1, outputs2, neg_outputs, neg_sample_weights=1.0):
    outputs1 = F.normalize(outputs1, dim=1)
    outputs2 = F.normalize(outputs2, dim=1)
    neg_outputs = F.normalize(neg_outputs, dim=1)

    true_aff = F.cosine_similarity(outputs1, outputs2)
    # neg_aff = outputs1.mm(neg_outputs.t())    
    neg_aff = F.cosine_similarity(outputs1, neg_outputs)
    true_labels = torch.ones(true_aff.shape)
    if torch.cuda.is_available():
        true_labels = true_labels.cuda()
        true_xent = sigmoid_cross_entropy_with_logits(labels=true_labels, logits=true_aff)
    neg_labels = torch.zeros(neg_aff.shape)
    if torch.cuda.is_available():
        neg_labels = neg_labels.cuda()
    neg_xent = sigmoid_cross_entropy_with_logits(labels=neg_labels, logits=neg_aff)
    loss = true_xent.sum() + neg_sample_weights * neg_xent.sum()
    return loss  

class SupervisedGraphSage(nn.Module):

    def __init__(self, num_classes, enc):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc
        self.xent = nn.CrossEntropyLoss()

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))
        init.xavier_uniform(self.weight)

    def forward(self, nodes):
        embeds = self.enc(nodes)
        scores = self.weight.mm(embeds)
        self.embeds, self.scores = embeds.t(), scores.t()
        return embeds.t(), scores.t()

    def loss(self, nodes, labels):
        _, scores = self.forward(nodes)
        return self.xent(scores, labels.squeeze())

    def unsup_loss(self, nodes1, nodes2, neg_nodes):
        embeds1, _ = self.forward(nodes1)
        embeds2, _ = self.forward(nodes2)
        neg_embeds, _ = self.forward(neg_nodes)
        loss = node2vec(embeds1, embeds2, neg_embeds)
        return loss 

def load_cora():
    num_nodes = 2708
    num_feats = 1433
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes,1), dtype=np.int64)
    node_map = {}
    label_map = {}
    with open("cora/cora.content") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            feat_data[i,:] = np.array(list(map(float, info[1:-1])))
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]

    adj_lists = defaultdict(set)
    adj = np.zeros((len(node_map), len(node_map)))
    edges = []
    with open("cora/cora.cites") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
            adj[paper1][paper2]=1
            adj[paper2][paper1]=1
            edges.append([paper1, paper2])
    return feat_data, labels, adj_lists, adj, adj.sum(1), np.array(edges)

def run_cora():
    np.random.seed(1)
    random.seed(1)
    num_nodes = 2708
    feat_data, labels, adj_lists, adj, degrees, edges = load_cora()
    neg_adj_list = [list(set(range(num_nodes)).intersection(adj_lists[node])) for node in range(num_nodes)]
    features = nn.Embedding(2708, 1433)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
    adj = Variable(torch.FloatTensor(adj), requires_grad=False)
    if torch.cuda.is_available():
        features, adj = features.cuda(), adj.cuda()
   # features.cuda()

    agg1 = MeanAggregator(features, cuda=True)
    enc1 = Encoder(features, 1433, 128, adj_lists, agg1, gcn=True, cuda=False)
    agg2 = MeanAggregator(lambda nodes : enc1(nodes).t(), cuda=False)
    enc2 = Encoder(lambda nodes : enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2,
            base_model=enc1, gcn=True, cuda=False)
    enc1.num_samples = 5
    enc2.num_samples = 5

    graphsage = SupervisedGraphSage(7, enc2)
    if torch.cuda.is_available():
        graphsage = graphsage.cuda()
    # rand_indices = np.random.permutation(num_nodes)
    # test = rand_indices[:1000]
    # val = rand_indices[1000:1500]
    # train = list(rand_indices[1500:])

    train = list(range(140))
    val = np.array(range(200, 500))
    test = np.array(range(500, 1500))

    optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, graphsage.parameters()), lr=0.7)

    times = []
    for batch in range(100):
        batch_edges = edges[:256]
        random.shuffle(batch_edges)
        nodes1, nodes2 = batch_edges[:, 0], batch_edges[:,1]
        # neg_nodes = fixed_unigram_candidate_sampler(
        #     num_sampled=NUM_SAMPLED,
        #     unique=False,
        #     range_max=len(degrees),
        #     distortion=0.75,
        #     unigrams=degrees
        # )
        neg_nodes = sample_negative(nodes1, neg_adj_list)   
        start_time = time.time()
        optimizer.zero_grad()
        loss = graphsage.unsup_loss(nodes1, nodes2, neg_nodes)
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time-start_time)
        if batch % 100 == 0:
            embeds,_ = graphsage.forward(list(range(num_nodes)))
            accs = classify(embeds.detach().cpu().numpy(),labels, 0.5)
            print(batch, loss.item(), accs)

    times = []
    for batch in range(100):
        batch_nodes = train[:256]
        random.shuffle(train)
        start_time = time.time()
        optimizer.zero_grad()
        batch_labels = Variable(torch.LongTensor(labels[np.array(batch_nodes)]))
        if torch.cuda.is_available():
            batch_labels = batch_labels.cuda()
        loss = graphsage.loss(batch_nodes, batch_labels)
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time-start_time)
        print (batch, loss.item())

    _,test_output = graphsage.forward(test) 
    print ("Testing F1:", f1_score(labels[test], test_output.detach().cpu().numpy().argmax(axis=1), average="micro"))
    print ("Average batch time:", np.mean(times))

def load_pubmed():
    #hardcoded for simplicity...
    num_nodes = 19717
    num_feats = 500
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes, 1), dtype=np.int64)
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
    np.random.seed(1)
    random.seed(1)
    num_nodes = 19717
    feat_data, labels, adj_lists = load_pubmed()
    features = nn.Embedding(19717, 500)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
   # features.cuda()

    agg1 = MeanAggregator(features, cuda=True)
    enc1 = Encoder(features, 500, 128, adj_lists, agg1, gcn=True, cuda=False)
    agg2 = MeanAggregator(lambda nodes : enc1(nodes).t(), cuda=False)
    enc2 = Encoder(lambda nodes : enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2,
            base_model=enc1, gcn=True, cuda=False)
    enc1.num_samples = 10
    enc2.num_samples = 25

    graphsage = SupervisedGraphSage(3, enc2)
#    graphsage.cuda()
    rand_indices = np.random.permutation(num_nodes)
    test = rand_indices[:1000]
    val = rand_indices[1000:1500]
    train = list(rand_indices[1500:])

    optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, graphsage.parameters()), lr=0.7)
    times = []
    for batch in range(200):
        batch_nodes = train[:1024]
        random.shuffle(train)
        start_time = time.time()
        optimizer.zero_grad()
        loss = graphsage.loss(batch_nodes, 
                Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time-start_time)
        print (batch, loss.data[0])

    val_output = graphsage.forward(val) 
    print ("Validation F1:", f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro"))
    print ("Average batch time:", np.mean(times))

if __name__ == "__main__":
    run_cora()
