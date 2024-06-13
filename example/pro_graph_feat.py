# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 15:46:03 2023

@author: dfhuang
"""

from prographUtils import pro_graphize
import pandas as pd
import numpy as np
import pickle
       
 ###########Test Set##################           
''' 蛋白质'''
df = pd.read_csv('./test-set-valid.csv')
pdbids = df['pdbs'].tolist()
pdbid_list = [i[:-1] for i in pdbids]

pro_graph = pro_graphize(pdbid_list, set_cate ="test")
pro_graph = np.reshape(pro_graph, (pro_graph.shape[0],1,pro_graph.shape[1],pro_graph.shape[2]))
        

with open('./test_prograph/test_prographfea.pkl','wb') as f:
    pickle.dump(pro_graph,f)