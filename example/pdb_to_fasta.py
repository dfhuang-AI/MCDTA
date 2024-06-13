# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 16:47:00 2023

@author: dfhuang
"""
import os
from Bio.PDB import PDBParser


three_to_one = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}
for pdb_file in os.listdir('./test_pro'):
    if pdb_file.endswith(".pdb"):
        pdbid = pdb_file[:4]

        fasta_file = './test_fasta/'+pdbid +".fasta"
    
        parser = PDBParser()
        
        try:
            structure = parser.get_structure("protein", './test_pro/'+pdb_file)
        
            model = structure[0]
        
            sequence = ""
            for chain in model:
                for residue in chain:
                    if residue.get_id()[0] == " ":
                        residue_name = three_to_one.get(residue.get_resname())
                        if residue_name:
                            sequence += residue_name
        
            with open(fasta_file, "w") as f:
                f.write(f">protein\n{sequence}\n")
        
            print("Conversion complete.")
        
        except Exception as e:
            print("Error:", e)