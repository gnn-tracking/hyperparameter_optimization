#!/usr/bin/env bash

#SBATCH --job-name=fixed-ec-test
#SBATCH --output=logs/fixed-ec-test-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:03:00
#SBATCH --cpus-per-task 1
#SBATCH --gres=gpu:1
#SBATCH --mem=70G

python3 ../scripts/tune_ec.py \
  --local \
  --only-enqueued \
  --enqueue "${1}" \
  --no-scheduler \
  --test
