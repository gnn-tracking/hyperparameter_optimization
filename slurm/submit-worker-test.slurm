#!/bin/bash

#SBATCH --job-name=ray-worker
#SBATCH --output=logs/ray-worker-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task 12
#SBATCH --gres=gpu:1
#SBATCH --mem=70G

ip_head=$(cat "${HOME}/.ray_head_ip_address")
echo "Using head node at '${ip_head}'"

redis_password=$(cat "${HOME}/.ray_head_redis_password")
echo "Redis password '${redis_password}'"

ray start \
  --address "$ip_head" \
  --num-cpus "${SLURM_CPUS_PER_TASK}" \
  --num-gpus "1" \
  --block \
  --redis-password="${redis_password}"
