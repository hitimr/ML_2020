import os
import platform

if platform.system() in ["Darwin", "Linux"]:
    SYSTEM_PATH_SEP = "/"
else: # platform.system() == "Windows":
    SYSTEM_PATH_SEP = "\\"

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__)) + f"{SYSTEM_PATH_SEP}"
MODEL_DIR = PROJECT_ROOT + SYSTEM_PATH_SEP + "models" + SYSTEM_PATH_SEP