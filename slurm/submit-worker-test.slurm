#!/bin/bash

#SBATCH --job-name=ray-worker
#SBATCH --output=logs/ray-worker-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task 12
#SBATCH --gres=gpu:1
#SBATCH --mem=70G

ip_head="172.17.2.18:6380"
echo "Using head node at '${ip_head}'"

redis_password="279b2a0a-7538-499c-9ea1-bbb003b09c9b"
echo "Redis password '${redis_password}'"

ray start \
  --address "$ip_head" \
  --num-cpus "${SLURM_CPUS_PER_TASK}" \
  --num-gpus "1" \
  --block \
  --redis-password="${redis_password}"
