import ray
from queue import Queue


class TaskGraph

class Task:
    def __init__(self, *, compute_fn, batch):
        self._compute_fn = compute_fn
        self._batch = batch

    def compute(self):
        output_batch = self._compute_fn(self._batch)
        return output_batch


class PipeWorker:
    def __init(self):
        self._task_queue = Queue()
        self._output_queue = Queue()

    def do_work(self):
        while True:
            task = self._task_queue.get()
            if not task:
                break
            try:
                activation = task.compute()
            except Exception:
                # TODO(Hao): do something else
                self._output_queue.put((False, None))
                continue
            self._output_queue.put((True, activation))

        self._output_queue.put((False, None))

    def put_into_task_queue(self, task):
        self._task_queue.append(task)

    @property
    def task_queue(self):
        return self._task_queue

    @property
    def output_queue(self):
        return self._output_queue


# TODO(Hao&Zhuohan): JAX-related computation...
class Cell:
    def __init__(self,
                 next_jaxpr, forward_func, backward_func):
        pass

    def forward(self, next_jaxpr, input):
        pass

    def backward(self):
        pass


class Pipeline:
    def __init__(self, *, cells, num_workers=2):
        if num_workers < 2:
            raise RuntimeError()
        self._num_workers = num_workers
        self.pipe_workers = None

        # Assume each cell is a compute func derived from jaxpr.
        self.cells = cells

        # create remote pipe workers
        self._create_workers(self._num_workers)

    def _create_workers(self, num_workers):
        remote_worker = ray.remote(num_cpus=1, num_gpus={192:168:0.1:gpu:0, gpu:1, gpu:2})(PipeWorker)
        self.pipe_workers = [remote_worker.remote()
                             for _ in range(num_workers)]

    def release_workers(self):
        pass

    def run(self, batches):
        b = len(batches) # number of micro batches
        c = self._num_workers # number of workers


        # job is a list of (batch_idx, worker_idx) pairs.
        forward_schedules = GpipeSchedule(b, c)
        for step, jobs in enumerate(forward_schedules):
            # print("Step {}: running jobs {}...".format(step, jobs))
            self.launch_job(jobs, batches)
        for step, jobs in enumerate(reversed(forward_schedules)):
            self.launch_job(jobs, ...)

    def launch_job(self, batches, jobs):
        """Launch a list of jobs for this pipe step."""
        for batch_idx, cell_idx in jobs:
            # propagate activations between cells
            # forward / backward
            # TODO(Hao): propagate activations between cells
            pass

def GpipeSchedule(b, c):
    """Copied from https://github.com/facebookresearch/fairscale/blob/master/fairscale/nn/pipe/pipeline.py

    # k: clock number
    #
    # k (i,j) (i,j) (i,j)
    # - ----- ----- -----
    # 0 (0,0)
    # 1 (1,0) (0,1)
    # 2 (2,0) (1,1) (0,2)
    # 3       (2,1) (1,2)
    # 4             (2,2)


    Args:
        b: # micro batches
        c: # cells
    """
    for k in range(b + c - 1):
        return [(k - j, j) for j in range(max(1 + k - b, 0), min(1 + k, c))]

    return TaskGraph(), schedule
