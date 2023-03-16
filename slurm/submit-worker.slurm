#!/usr/bin/env bash

# Usage: start-head.sh [ray head IP]
# If ray head is not provided, it is read from ~/.ray_head_ip_address

#SBATCH --job-name=ray-worker
#SBATCH --output=logs/ray-worker-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task 12
#SBATCH --gres=gpu:1
#SBATCH --mem=70G
#SBATCH --mail-type=fail
#SBATCH --mail-user=kl5675@princeton.edu

function stats {
  jobstats "${SLURM_JOB_ID}"
}
trap stats EXIT


ip_head=$1
if [ -z "$ip_head" ]; then
  ip_head=$(cat "${HOME}/.ray_head_ip_address")
fi
if [ -z "${ip_head}" ]; then
  >&2 echo "No ray head IP configured"
  exit 234
fi
echo "Using head node at ${ip_head}"

ray start \
  --address "$ip_head" \
  --num-cpus "${SLURM_CPUS_PER_TASK}" \
  --num-gpus "1" \
  --block
