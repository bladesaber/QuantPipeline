import multiprocessing
from typing import Callable, Dict, List
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import time


class ParallelProcess(object):
    def __init__(self, pool_num: int, callback_func: Callable):
        self.pool_num = pool_num
        self.callback_func = callback_func

    def start(self, params: List[Dict]) -> List[Dict]:
        with multiprocessing.Pool(processes=self.pool_num) as pool:
            results = list(tqdm(
                pool.imap(self.callback_func, params),
                total=len(params),
                desc="Processing with multiprocessing"
            ))
            return results


class ParallelThread(object):
    def __init__(self, thread_num: int, callback_func: Callable):
        self.thread_num = thread_num
        self.callback_func = callback_func

    def start(self, params: List[Dict]) -> List[Dict]:
        with ThreadPoolExecutor(max_workers=self.thread_num) as executor:
            futures = [executor.submit(self.callback_func, param) for param in params]
            results = []
            for future in tqdm(futures, total=len(params), desc="Processing with threading"):
                results.append(future.result())
            return results


class TestExample(object):
    def __init__(self):
        self.params = [{"a": 1, "b": 2}, {"a": 3, "b": 4}, {"a": 5, "b": 6}, {"a": 7, "b": 8}]

    def process_callback_func(self, param: Dict) -> Dict:
        time.sleep(1)
        result = param.copy()
        result['sum'] = result['a'] + result['b']
        return result

    def thread_callback_func(self, param: Dict) -> Dict:
        time.sleep(1)
        result = param.copy()
        result['sum'] = result['a'] + result['b']
        return result

    def example_process(self):
        time_start = time.time()
        print(f"Running ParallelProcess example..., start time: {time_start}")
        parallel_process = ParallelProcess(pool_num=4, callback_func=self.process_callback_func)
        results = parallel_process.start(self.params)
        print(f"Results: {results}, spend time: {time.time() - time_start}")

    def example_thread(self):
        time_start = time.time()
        print(f"Running ParallelThread example..., start time: {time_start}")
        parallel_thread = ParallelThread(thread_num=4, callback_func=self.thread_callback_func)
        results = parallel_thread.start(self.params)
        print(f"Results: {results}, spend time: {time.time() - time_start}")


if __name__ == '__main__':
    test_example = TestExample()
    test_example.example_process()
    test_example.example_thread()
