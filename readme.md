<div align="center">

# [DEPRECATED] GNN Tracking Hyperparameter Optimization (this will probably change completely very soon)

[![Documentation Status](https://readthedocs.org/projects/gnn-tracking-hpo/badge/?version=latest)](https://gnn-tracking-hpo.readthedocs.io/en/latest/?badge=latest)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/gnn-tracking/hyperparameter_optimization/main.svg)](https://results.pre-commit.ci/latest/github/gnn-tracking/hyperparameter_optimization/main)
[![Python package](https://github.com/gnn-tracking/hyperparameter_optimization/actions/workflows/test.yaml/badge.svg)](https://github.com/gnn-tracking/hyperparameter_optimization/actions/workflows/test.yaml)
[![Check Markdown links](https://github.com/gnn-tracking/hyperparameter_optimization/actions/workflows/check-links.yaml/badge.svg)](https://github.com/gnn-tracking/hyperparameter_optimization/actions/workflows/check-links.yaml)

<img width="1042" alt="Screenshot of wandb" src="https://user-images.githubusercontent.com/13602468/200128053-403ba2ac-7b52-4822-a34a-f154f38cb874.png">

</div>

This repository hosts submission scripts and framework for hyperparameter optimization
of the models defined in [the main library](https://github.com/gnn-tracking/gnn_tracking).
Part of this are fully parameterized versions of the models.

## Framework

* Uses [ray tune](https://docs.ray.io/en/latest/tune/index.html) as overarching
  framework. For deployment on [SLURM][] managed HPC nodes, ray workers are deployed
  as SLURM batch jobs (as further described [here][slurm-deployment])
* [Optuna](https://optuna.readthedocs.io/) is used to power the search
* Results are reported to [wandb/weights & biases](https://wandb.ai/)

## Setup

First, follow the instructions from [the main library](https://github.com/gnn-tracking/gnn_tracking)
to set up the conda environment and install the package

```bash
pip install -e .
git submodule update --init --recursive
```

## Get started

* Use or adapt one of the tuning scripts in `scripts/`

### Training with fixed parameters (no tuning)

## Other links

* [ray-tune-slurm-demo](https://github.com/klieret/ray-tune-slurm-demo/):
  Simple project to try out some of the capabilities of ray tune and wandb,
  especially with batch submission
* [wandb-osh](https://github.com/klieret/wandb-offline-sync-hook/): package to trigger
  wandb syncs on compute nodes without internet
* [additional stoppers for ray tune](https://github.com/klieret/ray-tune-stoppers-contrib.git):
  package with additional early stopping conditions for trials used in our
  HPO

[SLURM]: https://slurm.schedmd.com/documentation.html
[slurm-deployment]: https://github.com/klieret/ray-tune-slurm-demo/#option-2-head-node-and-worker-nodes
