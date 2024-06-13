# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 10:08:24 2023

@author: dfhuang
"""

import os
import torch
import torch.nn as nn
from torch.nn import Linear, Dropout
from torch_geometric.nn import GCNConv
from torch_geometric.nn.aggr import GraphMultisetTransformer
from MCDTA_utils import NN, Normalization
from typing import List

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
norm = Normalization()
class GridConvNet(nn.Module):
    def __init__(self, in_channel: int=36, embedding_dim=128,out_channel: int=256,
                 med_channel: List=[16, 32, 64, 128],kernel_size:List=[2,3,4,5,6], 
                 kernel_index:int=0,stride=1, padding=1):
        super(GridConvNet, self).__init__()

        self.in_channel = in_channel
        self.med_channel = med_channel
        self.out_channel = out_channel
        self.emb = nn.Linear(in_channel, embedding_dim)
        self.dropout = 0.1
        self.dropout = nn.Dropout(self.dropout)
        
        self.layers = nn.Sequential(
            nn.Conv3d(embedding_dim, self.med_channel[1], kernel_size=kernel_size[kernel_index], stride=stride, padding=padding, bias=False),
            nn.BatchNorm3d(self.med_channel[1]),
            nn.LeakyReLU(),
            nn.Conv3d(self.med_channel[1], self.med_channel[2], kernel_size=kernel_size[kernel_index], stride=stride, padding=padding, bias=False),
            nn.BatchNorm3d(self.med_channel[2]),
            nn.LeakyReLU(),
            nn.Conv3d(self.med_channel[2], self.med_channel[3], kernel_size=kernel_size[kernel_index], stride=stride, padding=padding, bias=False),
            nn.BatchNorm3d(self.med_channel[3]),
            nn.LeakyReLU(),
            nn.Conv3d(self.med_channel[3], self.out_channel, kernel_size=kernel_size[kernel_index], stride=stride, padding=padding, bias=False),
            nn.BatchNorm3d(self.out_channel),
            nn.AdaptiveMaxPool3d(1)
            
        )
        
    def forward(self, x):
        x = self.dropout(self.emb(x.to(device)))
        x = self.layers(x.permute(0,4,1,2,3)).view(-1, self.out_channel)#self.out_channel)
        return x 
 
class GraphConvnet(nn.Module):

    def __init__(self, feature_dim: int = 42, embedding_dim: int = 128, output_dim: int = 128,
                 n_conv_layers: int = 3, n_fc_layers: int = 2, 
                 hidden_dim: int = 32, node_hidden: int = 128,
                 fc_hidden: int = 128, transformer_hidden: int = 128, dropout: float = 0.1,seed: int =1234, *args, **kwargs):
        super(GraphConvnet,self).__init__()
        torch.manual_seed(seed)        
        # GCN layer(s)
        self.emb = nn.Linear(feature_dim, embedding_dim)
        self.conv_layers = torch.nn.ModuleList()
        self.conv_layers.append(GCNConv(embedding_dim, hidden_dim))
        for k in range(n_conv_layers-1):
            self.conv_layers.append(GCNConv(hidden_dim * 2**k, hidden_dim * 2**(k+1)))

        # Global pooling
        self.transformer = GraphMultisetTransformer(in_channels=node_hidden,
                                                    hidden_channels=transformer_hidden,
                                                    out_channels=fc_hidden,
                                                    num_heads=8)

        self.activate = nn.LeakyReLU()

        self.dropout = Dropout(dropout)
        
        self.flat = nn.Linear(fc_hidden, output_dim)

    def forward(self,data):
        h = self.emb(data.x.to(device).float())
        #### Normalization
        h = h.cpu().detach().numpy()
        h = norm.normalize(h)
        h = torch.tensor(h).float().to(device)
        h = self.dropout(h)
        # Conv layers
        for k in range(len(self.conv_layers)):
            h = self.activate(self.conv_layers[k](h, data.to(device).edge_index))
        
        # Global graph pooling with a transformer
        h = self.transformer(x=h, index=data.to(device).batch, edge_index=data.to(device).edge_index)
        # Apply a fully connected layer.
        h = self.flat(h)
        return h
    
class Flat_Model(nn.Module):
    def __init__(self, in_channel: int=1, out_channel: int=128, kernel_size=3, stride=1, padding=1):
        super(Flat_Model, self).__init__()

        self.in_channel = in_channel
        self.med_channel = [32, 64, 128]
        self.out_channel = out_channel
        
        self.layers = nn.Sequential(
            nn.Conv2d(in_channel, self.med_channel[0], kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(self.med_channel[0]),
            nn.LeakyReLU(),
            nn.Conv2d(self.med_channel[0], self.med_channel[1], kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(self.med_channel[1]),
            nn.LeakyReLU(),
            nn.Conv2d(self.med_channel[1], out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.AdaptiveMaxPool2d(1)
        )
        
    def forward(self, x):
        x = self.layers(x).view(-1, self.out_channel)
        return x
    
class Sequence_Model(nn.Module):
    def __init__(self, in_channel: int=64, embedding_channel: int=128, out_channel: int=128, kernel_size=3, stride=1, padding=1, relative_position=False, Heads=None, use_residue=False):
        super(Sequence_Model, self).__init__()
        self.in_channel = in_channel
        self.med_channel = [32, 64, 128]
        self.out_channel = out_channel
        self.residue_in_channel = 64
        self.dim = '1d'
        self.dropout = 0.1
        
        self.emb = nn.Linear(in_channel, embedding_channel)
        self.dropout = nn.Dropout(self.dropout)
        
        self.layers = nn.Sequential(
            nn.Conv1d(embedding_channel, self.med_channel[0], kernel_size, stride, padding, bias=False),
            nn.BatchNorm1d(self.med_channel[0]),
            nn.LeakyReLU(),
            nn.Conv1d(self.med_channel[0], self.med_channel[1], kernel_size, stride, padding, bias=False),
            nn.BatchNorm1d(self.med_channel[1]),
            nn.LeakyReLU(),
            nn.Conv1d(self.med_channel[1], out_channel, kernel_size, stride, padding, bias=False),
            nn.BatchNorm1d(out_channel),
            nn.AdaptiveMaxPool1d(1)
        )

    def forward(self, x):
        x = self.dropout(self.emb(x.to(device)))
        x = self.layers(x.permute(0, 2, 1)).view(-1, self.out_channel)
        return x
    
class Multi_model(nn.Module):
    def __init__(self, compound_sequence_channel, protein_sequence_channel, out_channel, n_output=1):
        super(Multi_model, self).__init__()
        self.embedding_dim = 128
        
        self.compound_sequence = Sequence_Model(in_channel = compound_sequence_channel, 
                                                embedding_channel = self.embedding_dim, 
                                                out_channel = out_channel, kernel_size=3, padding=1)
        
        self.protein_sequence = Sequence_Model(in_channel = protein_sequence_channel, 
                                               embedding_channel = self.embedding_dim,
                                               out_channel = out_channel, kernel_size=3, padding=1)
    
        self.compound_stru = GraphConvnet(feature_dim = 42, embedding_dim = self.embedding_dim, output_dim = out_channel )
        
        self.protein_stru = Flat_Model(in_channel = 1, out_channel = out_channel)
        
        self.com_pro_grid = GridConvNet(in_channel = 36, kernel_index=3,out_channel = 256)
        
        
        self.fc_input = 128 * 4 + 256
        
        self.onedlayers = nn.Sequential(
            nn.Linear(128*2, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1))
        
        self.twodlayers = nn.Sequential(
            nn.Linear(128*2, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1))
        
        self.threedlayers = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1))

            
        self.reg = nn.Linear(3, 1)   
            

    def forward(self, compound_sequence, 
                compound_graph, 
                protein_sequence, 
                protein_graph,
                compound_protein_grid):
        
        c_sequence_feature = self.compound_sequence(compound_sequence)
        c_graph_feature = self.compound_stru(compound_graph)

        p_sequence_feature = self.protein_sequence(protein_sequence)
        p_graph_feature = self.protein_stru(protein_graph)
        
        pli_feature= self.com_pro_grid(compound_protein_grid)
        
        sequence = self.onedlayers(torch.cat((c_sequence_feature,p_sequence_feature),dim=1))
        structure = self.twodlayers(torch.cat((c_graph_feature,p_graph_feature),dim=1))
        grid = self.threedlayers(pli_feature)

        x = self.reg(torch.cat((sequence,structure,grid),dim=1))
        return x
    

class mcDTA(NN):
    def __init__(self, mol_sequence_channel: int=64, pro_sequence_channel: int=20,
                 out_channel: int=128, n_output: int=1,  
                 dropout: float = 0.1,lr: float = 0.0001, 
                 save_path: str = '.',prefix = '',
                 epochs: int = 200, *args, **kwargs):
        super().__init__()

        self.model = Multi_model(mol_sequence_channel, pro_sequence_channel, 
                                 out_channel, n_output=n_output)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)        
        self.save_path = os.path.join(save_path, prefix +'--best_model_param.pkl')
        self.epochs = epochs
        self.name = 'mcDTA'
        # Move the whole model to the gpu
        self.model = self.model.to(self.device)