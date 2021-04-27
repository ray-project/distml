import os
import subprocess
from itertools import combinations


def combine(temp_list, n):
    '''根据n获得列表中的所有可能组合（n个元素为一组）'''
    temp_list2 = []
    for c in combinations(temp_list, n):
        temp_list2.append(c)
    return temp_list2


def find_conbine(cuda_list, num_cuda=1):
    end_list = []
    for i in range(len(list1)):
        end_list.extend(combine(list1, i))

    new_list = []
    for i in end_list:
        if len(i) == num_cuda:
            new_list.append(",".join(i))
    return new_list


def find_wored_device():
    # find the cuda devices without cuda unhandled error
    worked_cuda = []
    num_workers = 4
    for cuda_aval in find_conbine(list1, num_workers):
        cmd =  f"CUDA_VISIBLE_DEVICES={cuda_aval} NCCL_SHM_DISABLE=1 python mnist_jax_example.py --num-epochs 1 --num-workers {num_workers} --smoke-test True"
        print(f"running command `{cmd}`")
        res = os.system(cmd)
        if res == 0:
            worked_cuda.append(cuda_aval)
    print("avaliable cuda list", worked_cuda)

# =================================================================

default_epoches = 5
cuda_list = ['0', '1', '2', '3', '4', '5', '6', '7']
num_gpus = 12
address = "172.18.167.27:6379"

def run_mnist_ar_distributed():
    num_workers = "--num-workers {}"
    num_workers_candidate = range(2, len(cuda_list) + 1)

    models = "--model-name {}"
    models_candidate = ["resnet18", "resnet101"]

    cuda_aval = ",".join(cuda_list)

    for workers in num_workers_candidate:
        for model in models_candidate:
            cmd =  f"CUDA_VISIBLE_DEVICES={cuda_aval} NCCL_SHM_DISABLE=1 python mnist_jax_example.py --num-epochs {default_epoches} {num_workers.format(workers)} --trainer ar {models.format(model)} --address {address}"
            res = os.system(cmd)


def run_wiki_ar_distributed():
    num_workers = "--num-workers {}"
    num_workers_candidate = range(2, num_gpus + 1)

    cuda_aval = ",".join(cuda_list)

    for workers in num_workers_candidate:
        cmd =  f"CUDA_VISIBLE_DEVICES={cuda_aval} NCCL_SHM_DISABLE=1 python wiki_flax_example.py --num-epochs {default_epoches} {num_workers.format(workers)} --trainer ar --address {address}"
        res = os.system(cmd)


def run_mnist_ps_distributed(num_ps=1):
    num_workers = "--num-workers {}"
    num_workers_candidate = range(num_ps*2, num_gpus + 1)

    num_ps_str = f"--num-ps {num_ps}"

    models = "--model-name {}"
    models_candidate = ["resnet18", "resnet101"]

    cuda_aval = ",".join(cuda_list)

    for workers in num_workers_candidate:
        for model in models_candidate:
            cmd =  f"CUDA_VISIBLE_DEVICES={cuda_aval} NCCL_SHM_DISABLE=1 python mnist_jax_example.py --num-epochs {default_epoches} {num_workers.format(workers)} {num_ps_str} --trainer ps {models.format(model)} --address {address}"
            res = os.system(cmd)


def run_wiki_ps_distributed(num_ps=1):
    num_workers = "--num-workers {}"
    num_workers_candidate = range(num_ps*2, num_gpus + 1)

    num_ps_str = f"--num-ps {num_ps}"

    cuda_aval = ",".join(cuda_list)

    for workers in num_workers_candidate:
        cmd =  f"CUDA_VISIBLE_DEVICES={cuda_aval} NCCL_SHM_DISABLE=1 python wiki_flax_example.py --num-epochs {default_epoches} {num_workers.format(workers)} {num_ps_str} --trainer ps --address {address}"
        res = os.system(cmd)


def run_12gpus(num_gpus=4):
    num_ps_str = "--num-ps {}"

    num_workers = "--num-workers {}"
    workers = num_gpus

    models = "--model-name {}"
    models_candidate = ["resnet18", "resnet101"]

    # for model in ["resnet18", "resnet101"]:
    #     start_servers(workers)
    #     os.system(f"NCCL_SHM_DISABLE=1 python mnist_jax_example.py --num-epochs {default_epoches} {num_workers.format(workers)} --trainer ar {models.format(model)} --address {address}")
    

    # start_servers(workers)
    # os.system(f"NCCL_SHM_DISABLE=1 python wiki_flax_example.py --num-epochs 1 {num_workers.format(workers)} --address {address}")


    for model in ["resnet18", "resnet101"]:
        for i in range(1,3):
            start_servers(workers, i*2)
            os.system(f"NCCL_SHM_DISABLE=1 python mnist_jax_example.py --num-epochs {default_epoches} {num_workers.format(workers)} --trainer ps {num_ps_str.format(i*2)} {models.format(model)} --address {address}")
        

    # for i in range(1,3):
    #     num_server = i*2
    #     start_servers(workers, i*2)
    #     os.system(f"NCCL_SHM_DISABLE=1 python wiki_flax_example.py --num-epochs {default_epoches} {num_workers.format(workers)} --trainer ps {num_ps_str.format(num_server-1)} --address {address}")


def start_servers(num_workers, num_ps=0):
    os.system(f"bash -i start_local_head_node_n_server.sh {num_workers//2} {num_ps//2} {num_workers//2-num_ps//2}")
    os.system(f"bash -i start_remote_node_n_server.sh {num_workers//2} {num_ps//2} {num_workers//2-num_ps//2}")


if __name__ == "__main__":
    gpus_list = [12, 10, 8, 6, 4, 2]
    gpus_list = [4]
    for gpus in gpus_list:
        run_12gpus(gpus)
