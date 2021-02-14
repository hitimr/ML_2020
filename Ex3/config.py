import os
import platform

if platform.system() in ["Darwin", "Linux"]:
    SYSTEM_PATH_SEP = "/"
else: # platform.system() == "Windows":
    SYSTEM_PATH_SEP = "\\"

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__)) + f"{SYSTEM_PATH_SEP}"



# System Constants
MNIST_IMG_HEIGHT = 28
MNIST_IMG_HWIDTH = 28
MNIST_PIXEL_CNT = MNIST_IMG_HEIGHT * MNIST_IMG_HWIDTH