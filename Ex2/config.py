import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__)) + "\\"


DATASET_TEST_HOUSEPRICES = PROJECT_ROOT + "test\\test_datasets\\test_housePrices.csv"



FP_TOLERANCE = 1e-8 # Tolerance for floating point operations for unit tests