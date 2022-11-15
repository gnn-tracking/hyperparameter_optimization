<div align="center">

# GNN Tracking Hyperparameter Optimization

[![Documentation Status](https://readthedocs.org/projects/gnn-tracking-hpo/badge/?version=latest)](https://gnn-tracking-hpo.readthedocs.io/en/latest/?badge=latest)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/gnn-tracking/hyperparameter_optimization/main.svg)](https://results.pre-commit.ci/latest/github/gnn-tracking/hyperparameter_optimization/main)
[![Python package](https://github.com/gnn-tracking/hyperparameter_optimization/actions/workflows/test.yaml/badge.svg)](https://github.com/gnn-tracking/hyperparameter_optimization/actions/workflows/test.yaml)
[![Check Markdown links](https://github.com/gnn-tracking/hyperparameter_optimization/actions/workflows/check-links.yaml/badge.svg)](https://github.com/gnn-tracking/hyperparameter_optimization/actions/workflows/check-links.yaml)

<img width="1042" alt="Screenshot of wandb" src="https://user-images.githubusercontent.com/13602468/200128053-403ba2ac-7b52-4822-a34a-f154f38cb874.png">

</div>

This repository hosts submission scripts for hyperparameter optimization
related to [the main library](https://github.com/GageDeZoort/gnn_tracking).

## Framework

* Uses [ray tune](https://docs.ray.io/en/latest/tune/index.html) as overarching
  framework
* [Optuna](https://optuna.readthedocs.io/) is used to power the search
* Results are reported to [wandb/weights & biases](https://wandb.ai/)
* For synchronization with `wandb`, [wandb-osh](https://github.com/klieret/wandb-offline-sync-hook/) is used (the hooks are already included in the tuning script, you only need to start `wandb-osh` on the head node)

## Setup

1. Follow the instructions from [the main library](https://github.com/GageDeZoort/gnn_tracking)
   to set up the conda environment and install the package
2. `pip install .` this package.

## Get started

* Use or adapt one of the tuning scripts in `scripts/`

## Other links

* [ray-tune-slurm-test](https://github.com/klieret/ray-tune-slurm-test/):
  Simple project to try out some of the capabilities of ray tune and wandb,
  especially with batch submission
* [wandb-osh](https://github.com/klieret/wandb-offline-sync-hook/) package to trigger
  wandb syncs on compute nodes without internet
