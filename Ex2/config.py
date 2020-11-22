import os
import platform

if platform.system() in ["Darwin", "Linux"]:
    SYSTEM_PATH_SEP = "/"
else: # platform.system() == "Windows":
    SYSTEM_PATH_SEP = "\\"

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__)) + f"{SYSTEM_PATH_SEP}"


DATASET_TEST_HOUSEPRICES = PROJECT_ROOT + f"test{SYSTEM_PATH_SEP}test_datasets{SYSTEM_PATH_SEP}test_housePrices.csv"

DATASET_MONEYBALL = PROJECT_ROOT + "Datasets" + SYSTEM_PATH_SEP + "Moneyball" + SYSTEM_PATH_SEP + "baseball.csv"
MONEYBALL_TARGET = "W"
MONEYBALL_FEATURES = ["Team","League","Year","RS","RA","W","OBP","SLG","BA","Playoffs","RankSeason","RankPlayoffs","G","OOBP","OSLG"]


FP_TOLERANCE = 1e-10 # Tolerance for floating point operations for unit tests