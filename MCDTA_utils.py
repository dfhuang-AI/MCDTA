# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 10:22:55 2023

@author: dfhuang
"""
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch.utils.data import Dataset
from typing import List
import numpy as np
from numpy.typing import ArrayLike

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class NN:
    def __init__(self):

        self.train_losses = []
        self.val_losses = []
        self.epoch = 0
        self.epochs = 100
        self.save_path = None

        self.model = None
        self.device = None
        self.loss_fn = None
        self.optimizer = None

    def train(self, x_train: List, y_train: List[float], x_val: List =None, y_val: List[float] = None,
              early_stopping_patience: int = 20, epochs: int = None, print_every_n: int = 2
              ,reduce_every_n: int= 5, lr: float = 0.0001):
        if epochs is None:
            epochs = self.epochs
        
        mol_sequence_loader = numpy_loader(np.array(x_train[0]), y_train)
        mol_graph_loader = graphs_to_loader(x_train[1], y_train)
        pro_sequence_loader = numpy_loader(np.array(x_train[2]), y_train)
        pro_graph_loader = numpy_loader(np.array(x_train[3]), y_train)
        mol_pro_grid_loader = numpy_loader(np.array(x_train[4]),y_train)
        
        patience = None if early_stopping_patience is None else 0

        for epoch in range(epochs):
            
            if patience is not None and patience >= early_stopping_patience:

                if print_every_n < epochs:
                    print('Stopping training early')
                try:
                    with open(self.save_path, 'rb') as handle:
                        self.model.load_state_dict(torch.load(handle))
                        #self.model = pickle.load(handle)

                    #os.remove(self.save_path)
                except Warning:
                    print('Could not load best model, keeping the current weights instead')

                break

            # As long as the model is still improving, continue training
            else:

                loss = self._one_epoch(mol_sequence_loader,
                                       mol_graph_loader,
                                       pro_sequence_loader,
                                       pro_graph_loader,
                                       mol_pro_grid_loader
                                       )
                self.train_losses.append(loss.item())

                val_loss = 0
                if x_val is not None:
                    val_pred = self.predict(x_val)
                    val_loss = self.loss_fn(squeeze_if_needed(val_pred), torch.tensor(y_val))

                self.val_losses.append(val_loss.item())

                self.epoch += 1

                # Pickle model if its the best
                if val_loss <= min(self.val_losses):

                    with open(self.save_path, 'wb') as handle:
                        torch.save(self.model.state_dict(),handle)
                        
                    patience = 0
                else:
                    patience += 1

                if self.epoch % print_every_n == 0:
                    print(f"Epoch {self.epoch} | Train Loss {loss} | Val Loss {val_loss} | patience {patience}")

    def _one_epoch(self, mol_sequence_loader,mol_graph_loader,pro_sequence_loader,pro_graph_loader,mol_pro_grid_loader):
        """ Perform one forward pass of the train data through the model and perform backprop
        :param train_loader: Torch geometric data loader with training data
        :return: loss
        """
        # Enumerate over the data
        lambda_ = 0.001
        
        for mol_seq_batch, mol_gra_batch, pro_seq_batch, pro_gra_batch,mol_pro_grid_batch in zip(
               mol_sequence_loader,
               mol_graph_loader,
               pro_sequence_loader,
               pro_graph_loader,
               mol_pro_grid_loader
                ):
            y = mol_seq_batch[1].to(self.device)
            # Move batch to gpu
            mol_sequence = mol_seq_batch[0].to(self.device)
            mol_graph = mol_gra_batch
            pro_sequence = pro_seq_batch[0].to(self.device)
            pro_graph = pro_gra_batch[0].to(self.device)
            mol_pro_grid = mol_pro_grid_batch[0].to(self.device)
            
            # Reset gradients
            self.optimizer.zero_grad()

            # Forward pass
            y_hat = self.model(mol_sequence.float(),
                               mol_graph,
                               pro_sequence.float(),
                               pro_graph.float(),
                               mol_pro_grid.float()
                               )

            # Calculating the loss and gradients
            loss = self.loss_fn(squeeze_if_needed(y_hat), squeeze_if_needed(y))
            if not loss > 0:
                break
            # Add L2 regularization item
            l2_reg = torch.tensor(0.).to(self.device)
            for param in self.model.parameters():
                l2_reg += torch.norm(param)
            loss += lambda_ * l2_reg
            # Calculate gradients
            loss.backward()

            # Update weights
            self.optimizer.step()

        return loss

    def test(self, x_test: List, y_test: List[float]):
        
        mol_sequence_loader = numpy_loader(np.array(x_test[0]), y_test)
        mol_graph_loader = graphs_to_loader(x_test[1], y_test)
        pro_sequence_loader = numpy_loader(np.array(x_test[2]), y_test)
        pro_graph_loader = numpy_loader(np.array(x_test[3]), y_test)
        mol_pro_grid_loader = numpy_loader(np.array(x_test[4]),y_test)
        
        y_pred, y_true = [], []
        with torch.no_grad():
            for mol_seq_batch, mol_gra_batch, pro_seq_batch, pro_gra_batch,mol_pro_grid_batch in zip(
                    mol_sequence_loader,
                    mol_graph_loader,
                    pro_sequence_loader,
                    pro_graph_loader,
                    mol_pro_grid_loader
                    ):
                y = mol_seq_batch[1].to(self.device)
                # Move batch to gpu
                mol_sequence = mol_seq_batch[0].to(self.device)
                mol_graph = mol_gra_batch
                pro_sequence = pro_seq_batch[0].to(self.device)
                pro_graph = pro_gra_batch[0].to(self.device)
                mol_pro_grid = mol_pro_grid_batch[0].to(self.device)

                y_hat = self.model(mol_sequence.float(),
                                   mol_graph,
                                   pro_sequence.float(),
                                   pro_graph.float(),
                                   mol_pro_grid.float()
                                   )
                y_hat = squeeze_if_needed(y_hat).tolist()
                if type(y_hat) is list:
                    y_pred.extend(y_hat)
                    y_true.extend(squeeze_if_needed(y).tolist())
                else:
                    y_pred.append(y_hat)
                    y_true.append(squeeze_if_needed(y).tolist())

        return torch.tensor(y_pred), torch.tensor(y_true)

    def predict(self, x:List, batch_size: int = 32):
        mol_sequence_loader = numpy_loader(np.array(x[0]))
        mol_graph_loader = DataLoader(x[1], batch_size=batch_size, shuffle=False)
        pro_sequence_loader = numpy_loader(np.array(x[2]))
        pro_graph_loader = numpy_loader(np.array(x[3]))
        mol_pro_grid_loader = numpy_loader(np.array(x[4]))
        
        y_pred = []
        with torch.no_grad():
            for mol_seq_batch, mol_gra_batch, pro_seq_batch, pro_gra_batch,mol_pro_grid_batch in zip(
                    mol_sequence_loader,
                    mol_graph_loader,
                    pro_sequence_loader,
                    pro_graph_loader,
                    mol_pro_grid_loader
                    ):
                mol_sequence = mol_seq_batch[0].to(self.device)
                mol_graph = mol_gra_batch
                pro_sequence = pro_seq_batch[0].to(self.device)
                pro_graph = pro_gra_batch[0].to(self.device)
                mol_pro_grid = mol_pro_grid_batch[0].to(self.device)
                
                y_hat = self.model(mol_sequence.float(),
                                   mol_graph,
                                   pro_sequence.float(),
                                   pro_graph.float(),
                                   mol_pro_grid.float()
                                   )
                y_hat = squeeze_if_needed(y_hat).tolist()
                

                if type(y_hat) is list:
                    y_pred.extend(y_hat)
                else:
                    y_pred.append(y_hat)

        return torch.tensor(y_pred)

    def __repr__(self):
        return "Neural Network baseclass for NN taking numpy arrays"
    
class Normalization():
    def __init__(self):
        self.data = None
        
    def z_score_normalize(self,data):
        mean = np.nanmean(data, axis=0)
        std = np.nanstd(data, axis=0)

        np.seterr(divide='ignore',invalid='ignore')
        normalized_data = (data - mean) / std
        return normalized_data
    def normalize(self,data):
        
        normalized_train_features = self.z_score_normalize(data)
        missing_values_count = np.sum(np.isnan(normalized_train_features), axis=0)

        for i in range(len(missing_values_count)):
            if missing_values_count[i] == normalized_train_features.shape[0]:      
                normalized_train_features[:,i] = data[:,i]
                
            elif missing_values_count[i] < normalized_train_features.shape[0] and missing_values_count[i]!=0:

                missing_indices = np.isnan(normalized_train_features[:,i])
                mode = np.nanmean(normalized_train_features[:,i])
                normalized_train_features[:,i][missing_indices] = mode 
            else:
                pass
        return normalized_train_features

class NumpyDataset(Dataset):
    def __init__(self, x: ArrayLike, y: List[float] = None):
        """ Create a dataset for the ChemBerta transformer using a pretrained tokenizer """
        super().__init__()

        if y is None:
            y = [0]*len(x)
        self.y = torch.tensor(y).unsqueeze(1).to(device)
        self.x = torch.tensor(x).float().to(device)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.y)

def numpy_loader(x: ArrayLike, y: List[float] = None, batch_size: int = 32, shuffle: bool = False):
    dataset = NumpyDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def graphs_to_loader(x: List[Data], y: List[float], batch_size: int = 32, shuffle: bool = False):
    """ Turn a list of graph objects and a list of labels into a Dataloader """
    for graph, label in zip(x, y):
        graph.y = torch.tensor(label).to(device)
    return DataLoader(x, batch_size=batch_size, shuffle=shuffle)

def squeeze_if_needed(tensor):
    from torch import Tensor
    if len(tensor.shape) > 1 and tensor.shape[1] == 1 and type(tensor) is Tensor:
        tensor = tensor.squeeze()
    return tensor