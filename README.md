# MCDTA

MCDTA: A 4D tensor‑enhanced multi‑dimensional convolutional neural network for accurate prediction of protein–ligand binding affinity

## Requirements

[python](https://www.python.org/)==3.7.12

[rdkit](https://www.rdkit.org/)==2023.3.2

[numpy](https://numpy.org/)==1.21.6

[pandas](https://pandas.pydata.org/)==1.3.5

[biopython](https://biopython.org/)==1.81

[scipy](https://scipy.org/)==1.7.3

[torch](https://pytorch.org/)==1.13.1+cu117

[torch_geometric]([PyG Documentation — pytorch_geometric documentation (pytorch-geometric.readthedocs.io)](https://pytorch-geometric.readthedocs.io/en/latest/index.html))==2.1.0


## Example usage

* Due to the protein files are too large, we put them into "Releases" module. You need to download and copy them to './data/' folder or './example/' folder in advance.

### 1. Use our pre-trained model
In this section，we provide the test set and two external validation sets data, you can directly conduct the following command to run our pre-trained model and get the results on the sets. 

```bash
# Run the following command.
python test_pretrain.py -S XXX
# XXX represents test/csar/astex
```

### 2. Run on your datasets

In this section, you must provide .mol2 file of the ligand as well as .pdb file of the protein. We provide an example for data preparation and feature engineering based on the test set.

#### (1) Firstly, convert .pdb file into .fasta file by running the following command.

 ```bash
cd example/
python pdb_to_fasta.py

 ```

#### (2) Next, you need to keep ligands that could be properly read and converted into standard SMILES and corresponding proteins. You can get the pdbid of each complex, SMILES of each ligand, and corresponding binding affinity value by running the following command.

 ```bash
python canonical_smiles_generation.py

 ```

#### (3) Then, you need to keep complexes that could generate protein-ligand interaction representations. You can get the valid test set information and protein-ligand interaction grid by running the following command.

 ```bash
python gridFeaturize.py

 ```
#### (4) Then, you can choose to generate protein distance matrix by running the following command.

 ```bash
python pro_graph_feat.py

 ```
#### (5) Finally, when all the data is ready, you can copy all feature files into '../data/' folder and train your own model by running the following command.

 ```bash
cd ..
python train.py

 ```
