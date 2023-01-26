#!/usr/bin/env bash

port=6380
dashboard_port=8841

echo "Using port ${port} for the ray head node. Make sure this is unique"
echo "Using port ${dashboard_port} for the ray dashboard. Make sure this is unique"


redis_password=$(uuidgen)
echo "Redis password ${redis_password}"
# echo "${redis_password}" > "${HOME}/.ray_head_redis_password"

head_node_ip=$(hostname --ip-address)
echo "IP ray head: ${head_node_ip}"
# echo "${head_node_ip}:${port}" > "${HOME}/.ray_head_ip_address"

ray start \
  -vvv  \
  --head \
  --node-ip-address="$head_node_ip" \
  --port=$port \
  --num-cpus 1 \
  --num-gpus 0 \
  --block \
  --dashboard-host=0.0.0.0  \
  --dashboard-port=${dashboard_port} \
  --include-dashboard=false

rm -f "${HOME}/.ray_head_redis_password"
rm -f "${HOME}/.ray_head_ip_address"
