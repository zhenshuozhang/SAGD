"""
Deep Graph Infomax in DGL

References
----------
Papers: https://arxiv.org/abs/1809.10341
Author's code: https://github.com/PetarV-/DGI
"""

import torch
import torch.nn as nn
import math
from gcn import GCN
from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch import SGConv
import torch.nn.functional as F
import dgl.function as fn

class Encoder(nn.Module):
    def __init__(self, g, in_feats, n_hidden, n_layers, activation, dropout, gnn_encoder, k = 1):
        super(Encoder, self).__init__()
        self.g = g
        self.gnn_encoder = gnn_encoder
        if gnn_encoder == 'gcn':
            self.conv = GCN(g, in_feats, n_hidden, n_hidden, n_layers, activation, dropout)
        elif gnn_encoder == 'sgc':
            self.conv = SGConv(in_feats, n_hidden, k=10, cached=True)

    def forward(self, features, corrupt=False):
        if corrupt:
            perm = torch.randperm(self.g.number_of_nodes())
            features = features[perm]
        if self.gnn_encoder == 'gcn':
            features = self.conv(features)
        elif self.gnn_encoder == 'sgc':
            features = self.conv(self.g, features)
        return features


# class Discriminator(nn.Module):
#     def __init__(self, n_hidden):
#         super(Discriminator, self).__init__()
#         self.weight = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
#         self.reset_parameters()
#
#     def uniform(self, size, tensor):
#         bound = 1.0 / math.sqrt(size)
#         if tensor is not None:
#             tensor.data.uniform_(-bound, bound)
#
#     def reset_parameters(self):
#         size = self.weight.size(0)
#         self.uniform(size, self.weight)
#
#     def forward(self, features, summary):
#         features = torch.matmul(features, torch.matmul(self.weight, summary))
#         return features

class SAGD(nn.Module):
    def __init__(self, n_in, n_h, n_layers, activation, khop, dropout=0.0, op='cat', lam_train=False):
        super(SAGD, self).__init__()

        self.enc = MLP(in_feats=n_in, hid_feats=n_h, out_feats=n_h, n_layers=n_layers, khop=khop, act=activation, dropout=dropout)

        
        self.act = nn.PReLU() if activation == 'prelu' else activation
        self.lin = nn.Linear(n_h, n_h)
        self.proj = nn.Linear(n_h, 1)
        self.hop_counter = nn.Linear(n_h, khop)
        self.d_counter = nn.Linear(n_h, 2)
        self.op = op
        self.lam_train = lam_train
        self.lam = nn.Parameter(torch.FloatTensor(khop, 1, 1))
        torch.nn.init.xavier_uniform_(self.lam.data)

    def forward(self, seq1, seq2):
        lam = F.softmax(self.lam)       
        h_1 = self.enc(seq1)
        h_2 = self.enc(seq2)
        #d_logits_1 = 0
        #d_logits_2 = 0
        d_logits_1 = self.d_counter(h_1[-1] - h_1[0])
        d_logits_2 = self.d_counter(h_2[-1] - h_2[0])
        h_1 = self.act(h_1)
        h_2 = self.act(h_2)
        if self.op == 'cat':
            if self.lam_train:
                h_1 = (lam * h_1).view(-1, h_1.shape[-1])
                h_2 = (lam * h_2).view(-1, h_2.shape[-1])
            else:
                h_1 = (h_1).view(-1, h_1.shape[-1])
                h_2 = (h_2).view(-1, h_2.shape[-1])
        else:
            if self.lam_train:
                h_1 = (lam * h_1).sum(0)
                h_2 = (lam * h_2).sum(0)
            else:
                h_1 = (h_1).sum(0)
                h_2 = (h_2).sum(0)
        hop_logits_1 = self.hop_counter(h_1)
        hop_logits_2 = self.hop_counter(h_2)
        sc_1 = ((self.lin(h_1.squeeze(0))).sum(1)).unsqueeze(0)
        sc_2 = ((self.lin(h_2.squeeze(0))).sum(1)).unsqueeze(0)

        logits = torch.cat((sc_1, sc_2), 1)
        
        return logits, d_logits_1, d_logits_2, hop_logits_1, hop_logits_2
        
    def opt_h(self, seq1, seq2):
        lam = F.softmax(self.lam).detach()        
        h_1 = self.enc(seq1)
        h_2 = self.enc(seq2)
        h_1 = self.act(h_1)
        h_2 = self.act(h_2)
        if self.op == 'cat':
            h_1 = (lam * h_1).view(-1, h_1.shape[-1])
            h_2 = (lam * h_2).view(-1, h_2.shape[-1])
        else:
            h_1 = (lam * h_1).sum(0)
            h_2 = (lam * h_2).sum(0)
        sc_1 = ((self.lin(h_1.squeeze(0))).sum(1)).unsqueeze(0)
        sc_2 = ((self.lin(h_2.squeeze(0))).sum(1)).unsqueeze(0)

        logits = torch.cat((sc_1, sc_2), 1)

        return logits

    def opt_lam(self, seq1, seq2):
        lam = F.softmax(self.lam)
        h_1 = self.enc(seq1).detach()
        h_2 = self.enc(seq2).detach()
        h_1 = self.act(h_1)
        h_2 = self.act(h_2)
        if self.op == 'cat':
            h_1 = (lam * h_1).view(-1, h_1.shape[-1])
            h_2 = (lam * h_2).view(-1, h_2.shape[-1])
        else:
            h_1 = (lam * h_1).sum(0)
            h_2 = (lam * h_2).sum(0)
        sc_1 = ((self.lin(h_1.squeeze(0))).sum(1)).unsqueeze(0)
        sc_2 = ((self.lin(h_2.squeeze(0))).sum(1)).unsqueeze(0)

        logits = torch.cat((sc_1, sc_2), 1)

        return logits

    # Detach the return variables
    def embed(self, seq):
        h_1 = self.enc(seq)
        h_1 = self.act(h_1)
        return h_1
    
    def d_pred(self, seq, adj, sparse):
        h = self.enc(seq, adj, sparse)
        d_logits = self.d_counter(h)

        return d_logits

class MLP(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, n_layers, act, khop, dropout, bias=True):
        super(MLP, self).__init__()
        self.fc = nn.Linear(in_feats, out_feats, bias=bias)
        self.act = nn.PReLU() if act == 'prelu' else act
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_feats))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            #torch.nn.init.constant(m.weight.data, 0)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
    
    def forward(self, seq):
        h = seq
        h = self.fc(h)
        if self.bias is not None:
            h += self.bias
        #h = self.act(h)

        return h

class Classifier(nn.Module):
    def __init__(self, n_hidden, n_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(n_hidden, n_classes)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc.reset_parameters()

    def forward(self, features):
        features = self.fc(features)
        return torch.log_softmax(features, dim=-1)
