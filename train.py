# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 09:50:55 2023

@author: dfhuang
"""
from utils import smi_onehot,pro_onehot
from graphFeaturize import Featurizer
import pandas as pd
import numpy as np
import torch
import pickle
import time
from evaluate_metrics import *
import random

############### load corresponding features ##################

def select_random_indices(lst, proportion, seed):
    random.seed(seed)  # set the random seed
    num_elements = int(len(lst) * proportion)
    random_indices = random.sample(range(len(lst)), num_elements)
    return random_indices

def load_features(set_cate: str="train",proportion = 0.9):
    
    df = pd.read_csv('./data/%s-set-valid.csv'% set_cate)
    y = df['y'].tolist()
    
    '''Ligand'''
    smiles = df['smiles'].tolist()
    feat = Featurizer()
    
    mol_sequence = smi_onehot(smiles ,max_len =150)
    mol_graph = feat.graphs(smiles)
    
    '''Protein'''
    pdbids = df['pdbs'].tolist()
    pdbid_list = [i[:-1] for i in pdbids]
    
    pro_sequence = pro_onehot(pdbid_list, max_len=900, set_cate = set_cate)
    
    '''Protein Graph''' 
    pro_graph = None
    if set_cate == 'train':       
        for idx in range(1,11):
            with open('./data/%s_prograph/%s_prographfea_%d.pkl'% (set_cate,set_cate,idx),'rb') as f:
                    feature = pickle.load(f)
            #print(feature.shape)
            if pro_graph is None:
                pro_graph = feature
            else:
                pro_graph = np.concatenate([pro_graph,feature],axis = 0)
    else:
        with open('./data/%s_prograph/%s_prographfea.pkl'% (set_cate,set_cate),'rb') as f:
                feature = pickle.load(f)
        #print(feature.shape)
        if pro_graph is None:
            pro_graph = feature
        else:
            pro_graph = np.concatenate([pro_graph,feature],axis = 0)
        
    '''Interaction'''
    mol_pro_grid = None
    if set_cate == 'train':       
        for idx in range(1,11):
            with open('./data/%s_pli-grid/%s_grids_%d.pkl'% (set_cate,set_cate,idx),'rb') as f:
                    feature = pickle.load(f)
            #print(feature.shape)
            if mol_pro_grid is None:
                mol_pro_grid = feature
            else:
                mol_pro_grid = np.concatenate([mol_pro_grid,feature],axis = 0)
    else:
        with open('./data/%s_pli-grid/%s_grids.pkl'% (set_cate,set_cate),'rb') as f:
                feature = pickle.load(f)
        #print(feature.shape)
        if mol_pro_grid is None:
            mol_pro_grid = feature
        else:
            mol_pro_grid = np.concatenate([mol_pro_grid,feature],axis = 0)
      
    X = []
    X.append(mol_sequence)
    X.append(mol_graph)
    X.append(pro_sequence)
    X.append(pro_graph)
    X.append(mol_pro_grid)
    if set_cate!='train':### Test set or External validation set
        seed = 123  # random seed
        indices = select_random_indices(X[0], proportion, seed)
    
        assert len(X[0]) == len(y)
        X2= []
        X2.append(X[0][indices])#mol_sequence
        X2.append([X[1][i] for i in indices])#mol_graph
        X2.append(X[2][indices])#pro_sequence
        X2.append(X[3][indices])#pro_graph 
        X2.append(X[4][indices])#mol_pro_grid
        y2=[y[j] for j in indices]   
    
        return X2, y2
    else: # Training set
        seed = 123  #Random seed
        train_indices = select_random_indices(X[0], proportion, seed)
        val_indices = [x for x in list(range(len(X[0]))) if x not in train_indices]
    
        assert len(X[0]) == len(y)
        X1= []
        X1.append(X[0][train_indices])#mol_sequence
        X1.append([X[1][i] for i in train_indices])#mol_graph
        X1.append(X[2][train_indices])#pro_sequence
        X1.append(X[3][train_indices])#pro_graph 
        X1.append(X[4][train_indices])#mol_pro_grid
        y1=[y[j] for j in train_indices] 
        
        X2= []
        X2.append(X[0][val_indices])#mol_sequence
        X2.append([X[1][i] for i in val_indices])#mol_graph
        X2.append(X[2][val_indices])#pro_sequence
        X2.append(X[3][val_indices])#pro_graph 
        X2.append(X[4][val_indices])#mol_pro_grid
        y2=[y[j] for j in val_indices]
        
        return X1, y1, X2, y2


if __name__=='__main__':
    
    from MCDTA import mcDTA

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    mol_sequence_dim = 64
    protein_sequence_dim = 20
    X_train, y_train, X_val, y_val = load_features(set_cate='train',proportion=0.9)

    print('start training!')
    timestamp = time.strftime('%Y-%m-%dT%H:%M:%S')
    Model = mcDTA(mol_sequence_channel= mol_sequence_dim,
                pro_sequence_channel = protein_sequence_dim,
                out_channel=128, n_output=1,lr=0.0001,
                save_path = './models',prefix=timestamp)
    
    Model.train(X_train, y_train, x_val=X_val, y_val=y_val)
    print('training done!')