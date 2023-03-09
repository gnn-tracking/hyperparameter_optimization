#!/usr/bin/env bash

#SBATCH --job-name=fixed-ec-test
#SBATCH --output=logs/ec-test-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task 1
#SBATCH --gres=gpu:1
#SBATCH --mem=70G

python3 ../scripts/tune_ec.py \
  --local \
  --no-scheduler \
  --group test
