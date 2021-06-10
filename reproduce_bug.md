# UNHANDLED_CUDA_ERROR

## Single 
Run `python mnist_jax_example.py` with set `NCCL_SHM_DISABLE=0`.

AllReduce strategy case. `python mnist_jax_example.py`
        Traceback (most recent call last):
        File "mnist_jax_example.py", line 172, in <module>
            strategy.train()
        File "/data3/huangrunhui/proj2/distml/distml/strategy/allreduce_strategy.py", line 78, in train
            metric = self.data_parallel_group.train_batch()
        File "/data3/huangrunhui/proj2/distml/distml/strategy/allreduce_strategy.py", line 382, in train_batch
            loss_vals = ray.get(
        File "/data3/huangrunhui/proj2/tmp_ray/ray/python/ray/_private/client_mode_hook.py", line 48, in wrapper
            return func(*args, **kwargs)
        File "/data3/huangrunhui/proj2/tmp_ray/ray/python/ray/worker.py", line 1439, in get
            raise value.as_instanceof_cause()
        ray.exceptions.RayTaskError(NcclError): ray::Replica.train_batch() (pid=35265, ip=172.18.167.27)
        File "python/ray/_raylet.pyx", line 503, in ray._raylet.execute_task
        File "python/ray/_raylet.pyx", line 447, in ray._raylet.execute_task.function_executor
        File "/data3/huangrunhui/proj2/tmp_ray/ray/python/ray/_private/function_manager.py", line 566, in actor_method_executor
            return method(__ray_actor, *args, **kwargs)
        File "/data3/huangrunhui/proj2/distml/distml/strategy/allreduce_strategy.py", line 234, in train_batch
            col.allreduce(cg, self.group_name)
        File "/data3/huangrunhui/proj2/tmp_ray/ray/python/ray/util/collective/collective.py", line 262, in allreduce
            g.allreduce([tensor], opts)
        File "/data3/huangrunhui/proj2/tmp_ray/ray/python/ray/util/collective/collective_group/nccl_collective_group.py", line 184, in allreduce
            self._collective(tensors, tensors, collective_fn)
        File "/data3/huangrunhui/proj2/tmp_ray/ray/python/ray/util/collective/collective_group/nccl_collective_group.py", line 577, in _collective
            comms = self._get_nccl_collective_communicator(key, devices)
        File "/data3/huangrunhui/proj2/tmp_ray/ray/python/ray/util/collective/collective_group/nccl_collective_group.py", line 426, in _get_nccl_collective_communicator
            nccl_util.groupEnd()
        File "cupy/cuda/nccl.pyx", line 207, in cupy.cuda.nccl.groupEnd
        File "cupy/cuda/nccl.pyx", line 240, in cupy.cuda.nccl.groupEnd
        File "cupy/cuda/nccl.pyx", line 128, in cupy.cuda.nccl.check_status
        cupy.cuda.nccl.NcclError: NCCL_ERROR_UNHANDLED_CUDA_ERROR: unhandled cuda error


PS strategy case. `python mnist_jax_example.py --strategy ps --num-ps 2 --num-worker 4`

        2021-06-11 01:14:28,123 ERROR worker.py:77 -- Unhandled error (suppress with RAY_IGNORE_UNHANDLED_ERRORS=1): ray::PS.send_params() (pid=41878, ip=172.18.167.27)
        File "python/ray/_raylet.pyx", line 503, in ray._raylet.execute_task
        File "python/ray/_raylet.pyx", line 447, in ray._raylet.execute_task.function_executor
        File "/data3/huangrunhui/proj2/tmp_ray/ray/python/ray/_private/function_manager.py", line 566, in actor_method_executor
            return method(__ray_actor, *args, **kwargs)
        File "/data3/huangrunhui/proj2/distml/distml/strategy/ps_strategy.py", line 383, in send_params
            col.send(cv, dst_rank, group_name=self.group_name)
        File "/data3/huangrunhui/proj2/tmp_ray/ray/python/ray/util/collective/collective.py", line 536, in send
            g.send([tensor], opts)
        File "/data3/huangrunhui/proj2/tmp_ray/ray/python/ray/util/collective/collective_group/nccl_collective_group.py", line 355, in send
            self._point2point(tensors, p2p_fn, send_options.dst_rank,
        File "/data3/huangrunhui/proj2/tmp_ray/ray/python/ray/util/collective/collective_group/nccl_collective_group.py", line 626, in _point2point
            comms = self._get_nccl_p2p_communicator(comm_key, my_gpu_idx,
        File "/data3/huangrunhui/proj2/tmp_ray/ray/python/ray/util/collective/collective_group/nccl_collective_group.py", line 499, in _get_nccl_p2p_communicator
            comm = nccl_util.create_nccl_communicator(2, nccl_uid, my_p2p_rank)
        File "/data3/huangrunhui/proj2/tmp_ray/ray/python/ray/util/collective/collective_group/nccl_util.py", line 108, in create_nccl_communicator
            comm = NcclCommunicator(world_size, nccl_unique_id, rank)
        File "cupy/cuda/nccl.pyx", line 279, in cupy.cuda.nccl.NcclCommunicator.__init__
        File "cupy/cuda/nccl.pyx", line 128, in cupy.cuda.nccl.check_status
        cupy.cuda.nccl.NcclError: NCCL_ERROR_UNHANDLED_CUDA_ERROR: unhandled cuda error
