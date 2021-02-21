import os
import psutil
import pathlib
from time import strftime, gmtime, time
 
def get_gbyte(byte):
    return byte / 1024**3

def get_mbyte(byte):
    return byte / 1024**2

def get_kbyte(byte):
    return byte / 1024

def elapsed_since(start):
    return strftime("%H:%M:%S", gmtime(time() - start))
 
def _get_process_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss

def convert(func):
    def wrapper(*args, **kwargs):
        to = "MB"
        if "to" in kwargs.keys():
            to = kwargs.pop("to")
        if to=="kB":
            out = get_kbyte(func(*args, **kwargs)), to
            print("Out: ")
            print(out)
            return out
        if to=="MB":
            return get_mbyte(func(*args, **kwargs)), to
        if to=="GB":
            return get_gbyte(func(*args, **kwargs)), to
    return wrapper

@convert
def get_process_memory():
    """looks up the process memory

    wrapped version that allows for specification of units via the to parameter
    """
    return _get_process_memory()

def log_memory(file, **kwargs):
    path = pathlib.Path(file)
    mem = get_process_memory(**kwargs)
    to_write = f"{mem[0]}, {mem[1]}\n"
    mode = "a"
    if not path.is_file():
        mode = "w"
        to_write = f"memory, unit\n" + to_write
    with open(path, mode) as f:
        f.write(to_write)

def profile(func):
    def wrapper(*args, **kwargs):
        mem_before = _get_process_memory()
        start = time()
        result = func(*args, **kwargs)
        elapsed_time = elapsed_since(start)
        mem_after = _get_process_memory()
        print(f"{func.__name__}: memory before: {get_mbyte(mem_before):.2f}MB, after: {get_mbyte(mem_after):.2f}MB, consumed: {get_mbyte(mem_after - mem_before):.2f}MB; exec time: {elapsed_time:.6f}")
        return result
    return wrapper

def profile_tensor(func):
    def wrapper(iters=1, *args, **kwargs):
        mem_before = _get_process_memory()
        start = time()
        for i in range(iters):
            result = func(*args, **kwargs)
        elapsed_time = time() - start #elapsed_since(start)
        mem_after = _get_process_memory()
        profile = {
            "before": get_mbyte(mem_before),
            "after": get_mbyte(mem_after),
            "consumed": get_mbyte(mem_after - mem_before),
            "time": elapsed_time / iters,
            "total_time": elapsed_time
        }
        
        print(f"{func.__name__}: memory before: {get_mbyte(mem_before):.2f}MB, after: {get_mbyte(mem_after):.2f}MB, consumed: {get_mbyte(mem_after - mem_before):.2f}MB; exec time: {elapsed_time:.6f}")
        
        
        return result, profile
    return wrapper

@profile
def test():
    np.arange(0,100000)


if __name__ == "__main__":
    print("123")
    import numpy as np
    profile(np.arange(0,100000))

    test()

    print("Testing get process memory:")
    print(get_process_memory(to="MB"))

    test_file = "./tmp/mem_log_test.log"
    print(f"Testing log memory to {test_file}")
    log_memory(test_file)
    print("Now when passing a pathlib.Path")
    log_memory(pathlib.Path(test_file))
    