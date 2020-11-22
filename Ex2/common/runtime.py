import time

def runtime(func):
    """Wrapper for simple runtime measuring of functions (or methods).

    Let's see how I can best make the runtime available to the outside for saving...
    """
    def wrapper(*args, **kwargs):
        print(f"Timing {func.__name__}...", end=" ")
        start = time.time()
        returns = func(*args, **kwargs)
        end = time.time()
        diff = end - start
        print(f"Measured runtime: {diff:.5e}s")
        return returns
    return wrapper