"""mpc_helpers.py
Helper functions for MPC execution
"""

def mpc_check(func):
    """Gathers the return values and evaluates for execution success.
    """
    def wrapper(*args, **kwargs):
        process_results = func(*args, **kwargs)
        if not all(process_results):
            print("SUCCESS")
        else:
            print("FAILURE")
    return wrapper
