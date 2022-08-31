#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import DataLoader

import random
from gnn_tracking.graph_construction.graph_builder import GraphBuilder
from gnn_tracking.models.track_condensation_networks import GraphTCN

# set up a model and trainer
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

graph_builder = GraphBuilder(str(Path("~/data/gnn_tracking/point_clouds").expanduser()), str(Path("~/data/gnn_tracking/graphs").expanduser()), redo=False)
graph_builder.process(verbose=True, n=2)


from gnn_tracking.training.graph_tcn_trainer import GraphTCNTrainer

# use cuda (gpu) if possible, otherwise fallback to cpu
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(f'Utilizing {device}')

# use reference graph to get relevant dimensions
g = graph_builder.data_list[0]
node_indim = g.x.shape[1]
edge_indim = g.edge_attr.shape[1]
hc_outdim = 2 # output dim of latent space

# partition graphs into train, test, val splits
graphs = graph_builder.data_list
n_graphs = len(graphs)
rand_array = np.random.uniform(low=0, high=1, size=n_graphs)
train_graphs = [g for i, g in enumerate(graphs) if (rand_array<=0.7)[i]]
test_graphs = [g for i, g in enumerate(graphs) if ((rand_array>0.7) & (rand_array<=0.9))[i]]
val_graphs = [g for i, g in enumerate(graphs) if (rand_array>0.9)[i]]

# build graph loaders
params = {'batch_size': 1, 'shuffle': True, 'num_workers': 1}

train_loader = DataLoader(list(train_graphs), **params)

params = {'batch_size': 1, 'shuffle': False, 'num_workers': 2}
test_loader = DataLoader(list(test_graphs), **params)
val_loader = DataLoader(list(val_graphs), **params)
loaders = {'train': train_loader, 'test': test_loader,
           'val': val_loader}
print('Loader sizes:', [(k, len(v)) for k, v in loaders.items()])



# from gnn_tracking.utils.early_stopping import StopEarly
model = GraphTCN(node_indim, edge_indim, hc_outdim, hidden_dim=64)

trainer = GraphTCNTrainer(model=model, loaders=loaders, )

import warnings
warnings.filterwarnings('ignore')
trainer.train(epochs=2, max_batches=1)