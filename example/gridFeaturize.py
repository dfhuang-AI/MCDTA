# -*- coding: utf-8 -*-
"""
Created on Thu May  4 09:45:15 2023

@author: dfhuang
"""

import pandas as pd
import numpy as np
import pickle
from rdkit import Chem
from interaction_components.plinteraction import get_interactions
from interaction_components.utils import get_atom_coords

class Feature_extractor:
    def __init__(self, lig_obj, pro_obj):
        self.ligand = lig_obj
        self.protein = pro_obj    
        self.atom_codes = {}

        others = ([3,4,5,11,12,13]+list(range(19,32))+list(range(37,51))+list(range(55,84)))
        atom_types = [1,6,7,8,15,16,34,[9,17,35,53],others]

        for i, j in enumerate(atom_types):
            if type(j) is list:
                for k in j:
                    self.atom_codes[k] = i
                
            else:
                self.atom_codes[j] = i              
        
        self.sum_atom_types = len(atom_types)
    ### define the types of interactions
        self.interaction_types = ['hbond','hydrophobic',
                             'metal-ligand','pistacking',
                             'pication','saltbridge',
                             'halogen','waterbridge']
               
        self.interaction_codes = {}
        for i,j in enumerate(self.interaction_types):
            self.interaction_codes[j] = i
            
        ###calculate interactions
        self.result = get_interactions(self.protein, self.ligand)
        interactions = self.result.interactions        
        ###### Interactions #######
        'hbonds'
        self.hbonds_ldon = interactions.hbonds_ldon
        self.hbonds_pdon = interactions.hbonds_pdon
        'hydrophobic'
        self.hydrophobic_contacts = interactions.hydrophobic_contacts
        'metal-ligand'
        self.metal_complexes = interactions.metal_complexes
        'pistacking'
        self.pistacking = interactions.pistacking
        'pication'
        self.pication_laro = interactions.pication_laro
        self.pication_paro = interactions.pication_paro
        'saltbridge'
        self.saltbridge_lneg = interactions.saltbridge_lneg
        self.saltbridge_pneg = interactions.saltbridge_pneg
        'halogen'
        self.halogen = interactions.halogen_bonds
        'waterbridge'
        self.waterbridge = interactions.water_bridges
        
        
        self.interactions_dict = {'hbond':self.hbonds_ldon+self.hbonds_pdon, 
                             'hydrophobic':self.hydrophobic_contacts, 
                             'metal-ligand':self.metal_complexes, 
                             'pistacking':self.pistacking,
                             'pication':self.pication_laro+self.pication_paro, 
                             'saltbridge':self.saltbridge_lneg+self.saltbridge_pneg,
                             'halogen':self.halogen,
                             'waterbridge':self.waterbridge}
    
        self.sum_interaction_types = len(self.interaction_types)+1
        
    def encode(self, atomic_num, isprotein):
        encoding = np.zeros(self.sum_atom_types*2)
        if not isprotein:
            encoding[self.atom_codes[atomic_num]] = 1.0
        else:
            encoding[self.sum_atom_types+self.atom_codes[atomic_num]] = 1.0
        
        return encoding
    
    def get_idx(self, interaction_type):
    
        if len(self.interactions_dict[interaction_type])==0:
            #print('No %s found'% interaction_type)
            return False
       
        else:         
            if interaction_type == 'hbond':
                hbonds = self.interactions_dict[interaction_type]
                lig_orig_idxs, pro_orig_idxs=[],[]
                for hbond in hbonds:
                    if hbond.protisdon:#hbonds_pdon
                        lig_orig_idxs.extend([hbond.a_orig_idx])
                        pro_orig_idxs.extend([hbond.d_orig_idx])
                        pro_orig_idxs.extend([hbond.h_orig_idx])
                    else:#hbonds_ldon
                        pro_orig_idxs.extend([hbond.a_orig_idx])
                        lig_orig_idxs.extend([hbond.d_orig_idx])
                        lig_orig_idxs.extend([hbond.h_orig_idx])
                                
            elif interaction_type == 'hydrophobic':
                hydrophobics = self.interactions_dict[interaction_type]
                lig_orig_idxs, pro_orig_idxs=[],[]
                for hydrophobic in hydrophobics:
                    lig_orig_idxs.extend([hydrophobic.ligatom_orig_idx])
                    pro_orig_idxs.extend([hydrophobic.bsatom_orig_idx])
         
            elif interaction_type == 'metal-ligand':
                metals = self.interactions_dict[interaction_type]
                lig_orig_idxs, pro_orig_idxs=[],[]
                for metal in metals:
                    if metal.location in ['ligand','ligand water']:
                        pro_orig_idxs.extend([metal.metal_orig_idx])
                        lig_orig_idxs.extend([metal.target_orig_idx])
                    #elif metal.location in ['protein.sidechain','protein water']:
                        #pro_orig_idxs.extend([metal.target_orig_idx])
                        
                                 
            elif interaction_type == 'pistacking':
                pistackings = self.interactions_dict[interaction_type]
                lig_orig_idxs, pro_orig_idxs=[],[]
                for pistacking in pistackings:
                    lig_orig_idxs.extend(pistacking.ligatoms_orig_idx)
                    pro_orig_idxs.extend(pistacking.bsatoms_orig_idx)
            
            elif interaction_type == 'pication':
                pications = self.interactions_dict[interaction_type]
                lig_orig_idxs, pro_orig_idxs=[],[]
                for pication in pications:
                    if pication.protcharged:
                        lig_orig_idxs.extend(pication.ring_orig_idx)
                        pro_orig_idxs.extend(pication.charge_orig_idx)
                    else:
                        lig_orig_idxs.extend(pication.charge_orig_idx)
                        pro_orig_idxs.extend(pication.ring_orig_idx)
                                 
            elif interaction_type == 'saltbridge':
                saltbridges = self.interactions_dict[interaction_type]
                lig_orig_idxs, pro_orig_idxs=[],[]
                for saltbridge in saltbridges:
                    if saltbridge.protispos:
                        lig_orig_idxs.extend(saltbridge.neg_orig_idx)
                        pro_orig_idxs.extend(saltbridge.pos_orig_idx)
                    else:
                        lig_orig_idxs.extend(saltbridge.pos_orig_idx)
                        pro_orig_idxs.extend(saltbridge.neg_orig_idx)
                              
            elif interaction_type == 'halogen':
                halogens = self.interactions_dict[interaction_type]
                lig_orig_idxs, pro_orig_idxs=[],[]
                for halogen in halogens:
                    lig_orig_idxs.extend([halogen.don_orig_idx])
                    pro_orig_idxs.extend([halogen.acc_orig_idx])
            
            elif interaction_type == 'waterbridge':
                waterbridges = self.interactions_dict[interaction_type]
                lig_orig_idxs, pro_orig_idxs=[],[]
                for waterbridge in waterbridges:
                   if waterbridge.protisdon:
                       #donor
                       lig_orig_idxs.extend([waterbridge.a_orig_idx])
                       pro_orig_idxs.extend([waterbridge.d_orig_idx])
                       pro_orig_idxs.extend([waterbridge.h_orig_idx])
                   else:
                       #acceptor
                       pro_orig_idxs.extend([waterbridge.a_orig_idx])
                       lig_orig_idxs.extend([waterbridge.d_orig_idx])
                       lig_orig_idxs.extend([waterbridge.h_orig_idx])
                   if waterbridge.water_type == 'ligand':
                       lig_orig_idxs.extend([waterbridge.water_orig_idx])
                   else:
                       pro_orig_idxs.extend([waterbridge.water_orig_idx])
                       
            return lig_orig_idxs, pro_orig_idxs
    
    def get_features(self, molecule, isprotein):### molecule = ligand or protein; 
    
        coords = []
        features = [] 
                                 
        for atom in molecule.GetAtoms():
            coord = get_atom_coords(atom)
            atom_idx = atom.GetIdx()
            atom_type = atom.GetAtomicNum()
            
            atom_encoding = self.encode(atom_type,isprotein)
            inter_encoding = np.zeros(self.sum_interaction_types*2)
            if not isprotein:
                ##ligand
                for interaction_type in self.interaction_types:
                    if not self.get_idx(interaction_type):
                        continue
                    else:
                        lig_orig_idxs, pro_orig_idxs = self.get_idx(interaction_type)
                        inter_encoding[self.interaction_codes[interaction_type]] = 1.0*lig_orig_idxs.count(atom_idx)
    
                if np.max(inter_encoding) == 0.0:
                    inter_encoding[self.sum_interaction_types-1] = 1.0
                               
                encoding = np.concatenate([atom_encoding,inter_encoding],axis = 0)
                features.append(encoding)
            else:
                ##protein
                for interaction_type in  self.interaction_types:
                    if not self.get_idx(interaction_type):
                        continue
                    else:
                        lig_orig_idxs, pro_orig_idxs = self.get_idx(interaction_type)
                        inter_encoding[self.sum_interaction_types+self.interaction_codes[interaction_type]] = 1.0*pro_orig_idxs.count(atom_idx)
                
                if np.max(inter_encoding) == 0.0:
                    inter_encoding[(self.sum_interaction_types*2-1)] = 1.0
                    
                encoding = np.concatenate([atom_encoding,inter_encoding],axis = 0)   
                features.append(encoding)  
                
            coords.append(coord)
        coords = np.array(coords, dtype=np.float32)
        features = np.array(features, dtype=np.float32)
        
        return coords, features
    
    
    def grid(self,coords, features, resolution=1.0, max_dist=10.0):
        assert coords.shape[1] == 3
        assert coords.shape[0] == features.shape[0]  
      
        grid=np.zeros((1,20,20,20,features.shape[1]),dtype=np.float32)
        x=y=z=np.array(range(-10,10),dtype=np.float32)+0.5
        for i in range(len(coords)):
            coord=coords[i]
            tmpx=abs(coord[0]-x)
            tmpy=abs(coord[1]-y)
            tmpz=abs(coord[2]-z)
            if np.max(tmpx)<=19.5 and np.max(tmpy)<=19.5 and np.max(tmpz) <=19.5:
                grid[0,np.argmin(tmpx),np.argmin(tmpy),np.argmin(tmpz)] += features[i]
                    
        return grid                      
                    
def get_cnn_feature():
 
    test_pdb_ids = pd.read_csv('./test-set.csv')['pdbs'].tolist()      

    #################################################################################       
    #Feature engineering of test set. 
    test_grids,labels=None,[]
    valid_pbds,valid_smiles = [],[]
    for pdb_id in test_pdb_ids:
        #if core_id in pdbs:
        ligand = Chem.MolFromMol2File('./test_mol/'+pdb_id[:-1]+'_ligand.mol2', removeHs=False)  
        protein = Chem.MolFromPDBFile('./test_pro/'+pdb_id[:-1]+'_protein.pdb', removeHs=False)
        test_mol = Chem.MolFromMol2File('./test_mol/'+pdb_id[:-1]+'_ligand.mol2',removeHs=True)
        test_smile = Chem.MolToSmiles(test_mol)
        if "~" in test_smile:
            continue
        df1 = pd.read_csv('./test-set.csv')
        label = df1.loc[df1[(df1.pdbs == pdb_id)].index[0],'y']
        try:
            Feature = Feature_extractor(ligand,protein)
            coords1, features1 = Feature.get_features(ligand, False)
            coords2, features2 = Feature.get_features(protein, True)
        except:
            print(pdb_id[:-1],'can not get interactions')
            continue
        
        center=(np.max(coords1,axis=0)+np.min(coords1,axis=0))/2
        coords=np.concatenate([coords1,coords2],axis = 0)
        features=np.concatenate([features1,features2],axis = 0)
        assert len(coords) == len(features)
        coords = coords-center
        grid=Feature.grid(coords,features)
        if test_grids is None:
            test_grids = grid
        else:
            test_grids = np.concatenate([test_grids,grid],axis = 0)
        valid_pbds.append(pdb_id)
        valid_smiles.append(test_smile)
        labels.append(label)
    data = {'pdbs':valid_pbds,'smiles':valid_smiles,'y':labels}
    df = pd.DataFrame(data,columns=['pdbs','smiles','y'])
    df.to_csv('./test-set-valid.csv',index=False)
         
    test_label = np.array(labels,dtype=np.float32)
    with open('./test_pli-grid/test_label.pkl','wb') as f:
        pickle.dump(test_label,f)
    with open('./test_pli-grid/test_grids.pkl','wb') as f:
        pickle.dump(test_grids, f)
    print('test-set done!')     
    
if __name__=='__main__':
    get_cnn_feature()

