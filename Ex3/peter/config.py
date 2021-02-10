import os
import platform
import pathlib

if platform.system() in ["Darwin", "Linux"]:
    SYSTEM_PATH_SEP = "/"
else: # platform.system() == "Windows":
    SYSTEM_PATH_SEP = "\\"

# PETER_ROOT = os.path.dirname(os.path.abspath(__file__)) + f"{SYSTEM_PATH_SEP}"

PETER_ROOT = pathlib.Path(__file__).parent
DATA_DIR = PETER_ROOT / "data"

MNIST_SIZE = 60000


