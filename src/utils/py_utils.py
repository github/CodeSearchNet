import multiprocessing
from typing import List, Iterable, Callable, TypeVar


JobType = TypeVar("JobType")
ResultType = TypeVar("ResultType")


def __parallel_queue_worker(worker_id: int,
                            job_queue: multiprocessing.Queue,
                            result_queue: multiprocessing.Queue,
                            worker_fn: Callable[[int, JobType], Iterable[ResultType]]):
    while True:
        job = job_queue.get()

        # "None" is the signal for last job, put that back in for other workers and stop:
        if job is None:
            job_queue.put(job)
            break

        for result in worker_fn(worker_id, job):
            result_queue.put(result)
    result_queue.put(None)


def run_jobs_in_parallel(all_jobs: List[JobType],
                         worker_fn: Callable[[int, JobType], Iterable[ResultType]],
                         received_result_callback: Callable[[ResultType], None],
                         finished_callback: Callable[[], None],
                         result_queue_size: int=100) -> None:
    """
    Runs jobs in parallel and uses callbacks to collect results.
    :param all_jobs: Job descriptions; one at a time will be parsed into worker_fn.
    :param worker_fn: Worker function receiving a job; many copies may run in parallel.
      Can yield results, which will be processed (one at a time) by received_result_callback.
    :param received_result_callback: Called when a result was produced by any worker. Only one will run at a time.
    :param finished_callback: Called when all jobs have been processed.
    """
    job_queue = multiprocessing.Queue(len(all_jobs) + 1)
    for job in all_jobs:
        job_queue.put(job)
    job_queue.put(None)  # Marker that we are done

    # This will hold the actual results:
    result_queue = multiprocessing.Queue(result_queue_size)

    # Create workers:
    num_workers = multiprocessing.cpu_count() - 1
    workers = [multiprocessing.Process(target=__parallel_queue_worker,
                                       args=(worker_id, job_queue, result_queue, worker_fn))
               for worker_id in range(num_workers)]
    for worker in workers:
        worker.start()

    num_workers_finished = 0
    while True:
        result = result_queue.get()
        if result is None:
            num_workers_finished += 1
            if num_workers_finished == len(workers):
                finished_callback()
                break
        else:
            received_result_callback(result)

    for worker in workers:
        worker.join()
