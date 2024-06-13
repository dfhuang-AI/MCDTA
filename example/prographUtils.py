# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 15:57:44 2023

@author: dfhuang
"""
import math
import numpy as np
import cv2
from Bio.PDB.PDBParser import PDBParser

parser = PDBParser(PERMISSIVE=1,QUIET=True)

'''Generate and store distance matrix'''
def generate_dis_metirc(pdbid, set_cate):
    CA_Metric = []
    coordinate_list = []
    structure = parser.get_structure(pdbid, set_cate+ '_pro/'+ pdbid + '_protein.pdb')
    for chains in structure:
        for chain in chains:
            for residue in chain:
                for atom in residue:
                    if atom.get_name() == 'CA':
                        coordinate_list.append(list(atom.get_vector()))
    for i in range(len(coordinate_list)):
        ca_raw_list = []
        for j in range(len(coordinate_list)):
            if i == j:
                ca_raw_list.append(0)
            else:
                ca_raw_list.append(math.sqrt((coordinate_list[i][0]- coordinate_list[j][0]) ** 2 + (coordinate_list[i][1] - coordinate_list[j][1]) ** 2 + (coordinate_list[i][2] - coordinate_list[j][2]) ** 2))
        CA_Metric.append(ca_raw_list)
    return np.array(CA_Metric)

def pro_graphize(pdbid_list, set_cate: str="test"):
    feature_list = []
    for pdbid in pdbid_list:       
        img = generate_dis_metirc(pdbid, set_cate)
        img_resize = cv2.resize(img, (200, 200), interpolation = cv2.INTER_AREA)
        feature_list.append(img_resize)
    return np.array(feature_list, dtype=np.float64) #(Arry of float64: (n,200,200))
