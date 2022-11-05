# GNN Tracking Hyperparameter Optimization

This repository hosts submission scripts for hyperparameter optimization
related to [the main library](https://github.com/GageDeZoort/gnn_tracking).

## Framework

* Uses [ray tune](https://docs.ray.io/en/latest/tune/index.html) as overarching
  framework
* [Optuna](optuna.readthedocs.io/) is used to power the search
* Results are reported to [wandb/weights & biases](https://wandb.ai/)

## Setup

* You need [the main library](https://github.com/GageDeZoort/gnn_tracking)
* For synchronization with `wandb`, use [wandb-osh](https://github.com/klieret/wandb-offline-sync-hook/) (the hooks are already included in the tuning script, you only need to start `wandb-osh` on the head node)

## Get started

* Use or adapt one of the tuning scripts in `scripts/`

## Other links

* [ray-tune-slurm-test](https://github.com/klieret/ray-tune-slurm-test/):
  Simple project to try out some of the capabilities of ray tune and wandb,
  especially with batch submission
