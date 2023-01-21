#!/bin/bash

#SBATCH --job-name=fixed-ec
#SBATCH --output=logs/fixed-ec-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task 12
#SBATCH --gres=gpu:1

python3 ../scripts/tune_ec.py \
  --local \
  --gpu \
  --dname ec \
  --group ec \
  --timeout 1h50min \
  --only-enqueued \
  --enqueue $1 \
  --no-scheduler
