# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 15:25:15 2023

@author: dfhuang
"""
import os
import pandas as pd
from rdkit import Chem
#Get the valid data of affinties of proteins and ligands. -logKd/Ki values are the label.
affinity ={}
with open('./INDEX_general_PL_data.2020','r') as f:
    for line in f.readlines():
        if line[0] != '#':
            affinity[line.split()[0]] = line.split()[3]

test_pdbids = []
for file in os.listdir('./test_mol/'):
    test_pdbids.append(file[0:4])
        

## test set
test_pdbs, test_smiles, test_labels = [],[],[]               
for test_pdb in test_pdbids:
    try:
        ligand = Chem.MolFromMol2File('./test_mol/'+test_pdb+'_ligand.mol2',removeHs=False)  
        protein = Chem.MolFromPDBFile('./test_pro/'+test_pdb+'_protein.pdb',removeHs=False)     
        test_mol = Chem.MolFromMol2File('./test_mol/'+test_pdb+'_ligand.mol2',removeHs=True)
        test_smile = Chem.MolToSmiles(test_mol)
    except:
        continue
    test_smiles.append(test_smile)
    test_labels.append(affinity[test_pdb])
    test_pdbs.append(test_pdb+':') 
data = {'pdbs':test_pdbs,'smiles':test_smiles,'y':test_labels}
df = pd.DataFrame(data,columns=['pdbs','smiles','y'])
df.to_csv('./test-set.csv',index=False)


