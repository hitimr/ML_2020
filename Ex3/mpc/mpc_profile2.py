import cProfile
import functools
import pstats
import io

def profileit(func):
    @functools.wraps(func)  # <-- Changes here.
    def wrapper(*args, **kwargs):
        datafn = "tmp/" + func.__name__ + ".profile" # Name the data file sensibly
        prof = cProfile.Profile()
        retval = prof.runcall(func, *args, **kwargs)
        prof.dump_stats(datafn)
        return retval

    return wrapper

def profileit2(func):
    def wrapper(*args, **kwargs):
        datafn = func.__name__ + ".profile" # Name the data file sensibly
        prof = cProfile.Profile()
        retval = prof.runcall(func, *args, **kwargs)
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(prof, stream=s).sort_stats(sortby)
        ps.print_stats()
        with open(datafn, 'w') as perf_file:
            perf_file.write(s.getvalue())
        return retval

    return wrapper


TEST_SIZE = 100000
@profileit
def test():
    np.arange(1, TEST_SIZE)

if __name__ == "__main__":
    print("cProfile test")
    import numpy as np
    print("1: ")
    profileit(np.arange(0,TEST_SIZE))

    print("1: ")
    test()