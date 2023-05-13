#!/usr/bin/env bash

#SBATCH --job-name=fixed-ec
#SBATCH --output=logs/fixed-ec-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task 6
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --mail-type=fail
#SBATCH --mail-user=kl5675@princeton.edu

python3 ../scripts/tune_ec_large.py \
  --local \
  --group ec \
  --only-enqueued \
  --enqueue "${1}" \
  --dname ec-smaller \
  --tags ec-smaller-grid-search
