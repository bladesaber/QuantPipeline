import multiprocessing
from tqdm import tqdm
import time
from typing import List, Dict
from functools import partial
import numpy as np


def apply_async_run(pool_num, global_func, params: List, pack_params=None):
    if pack_params is None:
        pack_params = {}
    pool = multiprocessing.Pool(processes=pool_num)
    for param in params:
        pool.apply_async(
            partial(global_func, **pack_params), (param,)
        )
    pool.close()
    pool.join()


def apply_async_run_tqdm(pool_num, global_func, params: List, pack_params=None):
    if pack_params is None:
        pack_params = {}
    max_num = len(params)
    with multiprocessing.Pool(processes=pool_num) as pool:
        with tqdm(total=max_num) as pbar:
            for _ in pool.imap_unordered(partial(global_func, **pack_params), params):
                pbar.update()


def task(x):
    time.sleep(1)
    # print('ss:', x + 10.0)


if __name__ == '__main__':
    task_list = np.arange(0, 10000, 1).tolist()
    # apply_async_run(3, task, task_list)
    # apply_async_run_tqdm(3, task, task_list)
    pass
