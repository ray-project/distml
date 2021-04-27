#!/bin/bash
num_gpus=$1
num_server=$2
num_worker=$3
echo $num_gpus
MY_IPADDR=172.18.167.27
echo "connect remote node: 172.18.167.21"

ssh -o StrictHostKeyChecking=no huangrunhui@172.18.167.21 "source /data3/huangrunhui/anaconda3/etc/profile.d/conda.sh; conda activate distml; conda info --envs; ray stop; XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NCCL_SOCKET_IFNAME=enp11s0 NCCL_SHM_DISABLE=1 ray start --address='${MY_IPADDR}:6379' --num-gpus '${num_gpus}' --num-cpus 12 --object-manager-port=8076 --resources='{\"worker\":'${num_worker}', \"server\":'${num_server}'}'";

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 NCCL_SOCKET_IFNAME=enp11s0 ray start --address='172.18.167.27:6379' --object-manager-port=8076 --num-cpus 12 --num-gpus 6 --resources='{"server":1}' --resources='{"worker":"'${num_worker}'","server":"'${num_server}'"}'