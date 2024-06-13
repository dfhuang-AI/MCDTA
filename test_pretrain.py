# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 15:54:13 2023

@author: dfhuang
"""

import argparse
from utils import *
from evaluate_metrics import *
import pandas as pd
import torch
from MCDTA import mcDTA
from train import load_features

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mol_sequence_dim = 64
protein_sequence_dim = 20

# load corresponding features
def test(set_cate = 'test'):
    X_test, y_test = load_features(set_cate = set_cate,proportion=1.0) 
    ########################## load model ################################  
    with open ('./models/2024-03-25T14_23_47--best_model_param.pkl','rb') as handle:
        Model = mcDTA( mol_sequence_channel= mol_sequence_dim,
                    pro_sequence_channel = protein_sequence_dim,
                    out_channel=128, n_output=1)
        Model.model.load_state_dict(torch.load(handle,map_location=torch.device('cuda:0')))
        
    y_pred,y_true = Model.test(X_test, y_test)
    
    y_pred=y_pred.detach().numpy().flatten()
    y_true=y_true.detach().numpy().flatten()
    
    assert len(y_pred) == len(y_true)
    
    test_result = [mae(y_true, y_pred), rmse(y_true, y_pred), pearson(y_true, y_pred), spearman(y_true, y_pred), ci(y_true, y_pred), r_squared(y_true, y_pred)]
    print(test_result)
    
    #### Store data
    y_pred = [round(i,2) for i in y_pred]
    y_true = [round(i,2) for i in y_true]
    dic = {'y_pred':y_pred,'y_true':y_true}   
    df = pd.DataFrame(dic)
    df.to_excel('./results/%s_result.xlsx'%set_cate,index=False)
    return

def func():
    parser = argparse.ArgumentParser(description='parse dataset categories parameters')
    parser.add_argument('--Set','-S' ,type=str, help='the dataset name')
    args = parser.parse_args()
    set_cate = args.Set
    test(set_cate=set_cate)

if __name__ == "__main__":
    func()