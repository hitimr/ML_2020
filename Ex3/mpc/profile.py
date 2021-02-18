import os
import psutil
from time import strftime, gmtime, time
 
def elapsed_since(start):
    return strftime("%H:%M:%S", gmtime(time() - start))
 
 
def get_process_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss

def get_gbyte(byte):
    return byte / 1024**3

def get_mbyte(byte):
    return byte / 1024**2

def get_kbyte(byte):
    return byte / 1024
 
 
def profile(func):
    def wrapper(*args, **kwargs):
        mem_before = get_process_memory()
        start = time()
        result = func(*args, **kwargs)
        elapsed_time = elapsed_since(start)
        mem_after = get_process_memory()
        print(f"{func.__name__}: memory before: {get_mbyte(mem_before):.2f}MB, after: {get_mbyte(mem_after):.2f}MB, consumed: {get_mbyte(mem_after - mem_before):.2f}MB; exec time: {elapsed_time}")
        return result
    return wrapper

@profile
def test():
    np.arange(0,100000)

if __name__ == "__main__":
    print("123")
    import numpy as np
    profile(np.arange(0,100000))

    test()
    