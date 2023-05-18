#!/usr/bin/env bash

#SBATCH --job-name=fixed-ec
#SBATCH --output=logs/fixed-tcn-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task 6
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --mail-type=fail
#SBATCH --mail-user=kl5675@princeton.edu

python3 ../scripts/tune_tcn.py \
  --local \
  --group small-tcn \
  --only-enqueued \
  --enqueue "${1}" \
  --dname small-tcn
