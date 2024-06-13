# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 15:57:44 2023

@author: dfhuang
"""
import math
import numpy as np
import cv2
from Bio import SeqIO
from Bio.PDB.PDBParser import PDBParser

parser = PDBParser(PERMISSIVE=1,QUIET=True)

smiles_dict = {"#": 28, "%": 29, ")": 30, "(": 0, "+": 31, "-": 32, "/": 33, ".": 1, 
				"1": 34, "0": 2, "3": 35, "2": 3, "5": 36, "4": 4, "7": 37, "6": 5, 
				"9": 38, "8": 6, "=": 39, "A": 40, "@": 7, "C": 41, "B": 8, "E": 42, 
				"D": 9, "G": 43, "F": 10, "I": 44, "H": 11, "K": 45, "M": 46, "L": 12, 
				"O": 47, "N": 13, "P": 14, "S": 48, "R": 15, "U": 49, "T": 16, "W": 50, 
				"V": 17, "Y": 51, "[": 52, "Z": 18, "]": 53, "\\": 19, "a": 54, "c": 55, 
				"b": 20, "e": 56, "d": 21, "g": 57, "f": 22, "i": 58, "h": 23, "m": 59, 
				"l": 24, "o": 60, "n": 25, "s": 61, "r": 26, "u": 62, "t": 27, "y": 63}


amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

######################## mol feature ################################
def smiles_onehot(smi):
    seq = smi.split('\t')[0]
    integer_encoder = []
    onehot_encoder = []
    for item in seq:
        integer_encoder.append(smiles_dict[item])
    for index in integer_encoder:
        temp = [0 for _ in range(len(smiles_dict))]
        temp[index] = 1
        onehot_encoder.append(temp)
    return onehot_encoder

'''Input: SMILES list'''
def smi_onehot(smi_list,max_len: int =150):
    feature_list = []
    for smi in smi_list:       
        if max_len == 150:
            feature = smiles_onehot(smi)
            if len(feature) > 150:
                feature = feature[:150]
            feature_list.append(feature)
        else:
            print('max length error!')
    for i in range(len(feature_list)):
        if len(feature_list[i]) != max_len:
            for j in range(max_len - len(feature_list[i])):
                if max_len == 150:
                    temp = [0] * 64
                feature_list[i].append(temp)
    return np.array(feature_list, dtype=np.float64)#(Arry of float64: (n,150,64))

######################## Protein feature ############################
'''Generate and store distance matrix'''
def generate_dis_metirc(pdbid, set_cate):
    CA_Metric = []
    coordinate_list = []
    structure = parser.get_structure(pdbid, './data/'+set_cate+ '_pro/'+ pdbid + '_protein.pdb')
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

def pro_graphize(pdbid_list, set_cate: str="train"):
    feature_list = []
    for pdbid in pdbid_list:       
        img = generate_dis_metirc(pdbid, set_cate)
        img_resize = cv2.resize(img, (200, 200), interpolation = cv2.INTER_AREA)
        feature_list.append(img_resize)
    return np.array(feature_list, dtype=np.float64) #(Arry of float64: (n,200,200))
'''Input: single protein'''
def protein_onehot(pdbid, set_cate):
    for seq_recoder in SeqIO.parse('./data/'+set_cate + '_fasta/' + pdbid + '.fasta', 'fasta'):
        seq = seq_recoder.seq
    protein_to_int = dict((c, i) for i, c in enumerate(amino_acids))
    integer_encoded = [protein_to_int[char] for char in seq]
    onehot_encoded = []
    for value in integer_encoded:
        letter = [0 for _ in range(len(amino_acids))]
        letter[value] = 1
        onehot_encoded.append(letter)
    return onehot_encoded

'''Input: protein pdbid list'''
def pro_onehot(pdbid_list, max_len: int=900, set_cate: str="train"):
    feature_list = []
    for pdbid in pdbid_list:
        if max_len == 900:
            feature = protein_onehot(pdbid,set_cate)
            if len(feature) > 900:
                feature = feature[:900]
            feature_list.append(feature)
        else:
            print('max length error!')
    for i in range(len(feature_list)):
        if len(feature_list[i]) != max_len:
            for j in range(max_len - len(feature_list[i])):
                if max_len == 900:
                    temp = [0] * 20
                feature_list[i].append(temp)
    return np.array(feature_list, dtype=np.float64)#(Arry of float64: (n,900,20))
#############################################################################