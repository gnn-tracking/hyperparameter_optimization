#!/usr/bin/env bash

#SBATCH --job-name=mem-test
#SBATCH --output=logs/mem-test-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=70G
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=kl5675@princeton.edu
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1

python3 memory_test.py
