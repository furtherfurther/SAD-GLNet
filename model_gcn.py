import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.nn import RGCNConv, GraphConv, FAConv
from torch.nn import Parameter
import numpy as np, itertools, random, copy, math
import math
import scipy.sparse as sp
# import ipdb
# from HypergraphConv import HypergraphConv
from torch_geometric.nn import GCNConv
from itertools import permutations
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, global_add_pool as gsp
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import add_self_loops
from graphgcn import GraphGCN



class GraphConvolution(nn.Module):
    """
    Implements common graph convolution operation with special features such as variant mode and residual connection.
    """
    def __init__(self, in_features, out_features, residual=False, variant=False):
        super(GraphConvolution, self).__init__()
        # If using variant mode, double the input feature dimension
        self.variant = variant
        if self.variant:
            self.in_features = 2 * in_features 
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        # Weight parameter: input feature size x output feature size
        self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        # Initialize weight parameters
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights uniformly
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)



class GCN(nn.Module):
    def __init__(self, n_dim, nhidden, dropout, lamda, alpha, variant, return_feature, use_residue, 
                new_graph='full', n_speakers=2, modals=['a', 'v', 'l'], use_speaker=True, use_modal=False, num_L=3, num_K=4):
        super(GCN, self).__init__()
        self.return_feature = return_feature  # Whether to return features
        self.use_residue = use_residue
        self.new_graph = new_graph  # Whether to use new graph structure

        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda

        # Multimodal input configuration: audio, video, language
        self.modals = modals
        self.modal_embeddings = nn.Embedding(3, n_dim)  # Embedding for modalities
        self.speaker_embeddings = nn.Embedding(n_speakers, n_dim)  # Embedding for speakers
        self.use_speaker = use_speaker  # Whether to use speaker embedding
        self.use_modal = use_modal  # Whether to use modality embedding
        self.use_position = False  # Position encoding currently not used

        # Fully connected layer in network
        self.fc1 = nn.Linear(n_dim, nhidden)  # Mapping from input dimension to hidden layer



        return out1



