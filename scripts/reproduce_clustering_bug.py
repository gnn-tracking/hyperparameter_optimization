#!/usr/bin/env python3


from __future__ import annotations

from pathlib import Path

import torch
from gnn_tracking.graph_construction.graph_builder import GraphBuilder
from gnn_tracking.models.track_condensation_networks import GraphTCN
from gnn_tracking.postprocessing.dbscanscanner import dbscan_scan
from numpy.random import uniform
from torch_geometric.loader import DataLoader

torch.manual_seed(0)
import numpy as np

np.random.seed(0)
import random

random.seed(0)
# we'll use n_evts * n_sectors = 640 graphs
n_evts, n_sectors = 10, 64
indir = "/tigress/jdezoort/codalab/train_1"
# indir='/home/kl5675/Documents/22/git_sync/gnn_tracking/src/gnn_tracking/test_data'
# event_plotter = EventPlotter(indir=indir)
# event_plotter.plot_ep_rv_uv(evtid=21289)

# we can build graphs on the point clouds using geometric cuts


graph_builder = GraphBuilder(
    str(Path("~/data/gnn_tracking/point_clouds").expanduser()),
    str(Path("~/data/gnn_tracking/graphs").expanduser()),
    redo=False,
)
graph_builder.process(n=None)

from gnn_tracking.training.graph_tcn_trainer import GraphTCNTrainer

# use cuda (gpu) if possible, otherwise fallback to cpu
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(f"Utilizing {device}")

# use reference graph to get relevant dimensions
g = graph_builder.data_list[0]
node_indim = g.x.shape[1]
edge_indim = g.edge_attr.shape[1]
hc_outdim = 2  # output dim of latent space

# partition graphs into train, test, val splits
graphs = graph_builder.data_list
n_graphs = len(graphs)
rand_array = uniform(low=0, high=1, size=n_graphs)
train_graphs = [g for i, g in enumerate(graphs) if (rand_array <= 0.7)[i]]
test_graphs = [
    g for i, g in enumerate(graphs) if ((rand_array > 0.7) & (rand_array <= 0.9))[i]
]
val_graphs = [g for i, g in enumerate(graphs) if (rand_array > 0.9)[i]]

# build graph loaders
params = {"batch_size": 1, "shuffle": True, "num_workers": 1}


train_loader = DataLoader(list(train_graphs), **params)

params = {"batch_size": 2, "shuffle": False, "num_workers": 2}
test_loader = DataLoader(list(test_graphs), **params)
val_loader = DataLoader(list(val_graphs), **params)
loaders = {"train": train_loader, "test": test_loader, "val": val_loader}
print("Loader sizes:", [(k, len(v)) for k, v in loaders.items()])

# set up a model and trainer


import optuna
from gnn_tracking.training.tcn_trainer import TCNTrainer
from gnn_tracking.utils.losses import BackgroundLoss, EdgeWeightBCELoss, PotentialLoss

# optuna.logging.set_verbosity(optuna.logging.WARNING)

q_min, sb = 0.01, 0.1
loss_functions = {
    "edge": EdgeWeightBCELoss().to(device),
    "potential": PotentialLoss(q_min=q_min, device=device),
    "background": BackgroundLoss(device=device, sb=sb),
    # "object": ObjectLoss(device=device, mode='efficiency')
}

loss_weights = {
    # everything that's not mentioned here will be 1
    "edge": 5,
    "potential_attractive": 10,
    "potential_repulsive": 1,
    "background": 1,
    # "object": 1/250000,
}

# set up a model and trainer
model = GraphTCN(node_indim, edge_indim, hc_outdim, hidden_dim=64)
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
n_params = sum(np.prod(p.size()) for p in model_parameters)
print("number trainable params:", n_params)


# checkpoint = torch.load(Path("~/data/gnn_tracking/model.pt").expanduser())
# model.load_state_dict(checkpoint["model_state_dict"])


trainer = TCNTrainer(
    model=model,
    loaders=loaders,
    loss_functions=loss_functions,
    lr=0.0001,
    loss_weights=loss_weights,
    device=device,
    cluster_functions={"dbscan": dbscan_scan},
)
print(trainer.loss_functions)

# trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
# trainer._epoch = checkpoint["epoch"]


import warnings

warnings.filterwarnings("ignore")
trainer.test_step()
