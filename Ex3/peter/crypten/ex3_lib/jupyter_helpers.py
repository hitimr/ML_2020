"""jupyter_helpers.py

"""
class StopExecution(Exception):
    def _render_traceback_(self):
        pass

def stop_cell():
    print(("Stopping cell execution"))
    raise StopExecution
    print("This shouldn't appear...")
