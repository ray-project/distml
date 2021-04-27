num_gpus=$1
num_server=$2
num_worker=$3

ray stop
conda activate distml && NCCL_SOCKET_IFNAME=eno1 NCCL_SHM_DISABLE=1 ray start --head --object-manager-port=8076 --num-cpus=12 --num-gpus=$num_gpus --resources='{"worker":'${num_worker}',"server":'${num_server}'}'