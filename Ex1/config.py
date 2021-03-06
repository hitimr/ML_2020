import os

# Directories
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
COMPANY_DATA_DIR = WORKSPACE_DIR + "/datasets/companies"
BREASTCANCER_DATA_DIR = WORKSPACE_DIR + "/datasets/breast_cancer"
AMAZON_DATA_DIR = WORKSPACE_DIR + "/datasets/Amazon_Review_Data"
CONGRESSIONAL_VOTING_DIR = WORKSPACE_DIR + "/datasets/Congressional_Voting"
HEART_DISEASE_DIR = WORKSPACE_DIR + "/datasets/heart"


MIN_COMPANY_YEAR = 1
MAX_COMPANY_YEAR = 5

# Dataset sizes for unit tests
BREASTCANCER_LRN_SIZE = 9120
BREASTCANCER_SOL_SIZE = 568
BREASTCANCER_TES_SIZE = 8804

AMAZON_LRN_SIZE = 7501500
AMAZON_SOL_SIZE = 1500
AMAZON_TES_SIZE = 7500750

CONGRESSIONALVOTING_SAMPLE_SIZE = 434
CONGRESSIONALVOTING_TEST_SIZE = 3689
CONGRESSIONALVOTING_TRAIN_SIZE = 3924

HEART_DISEASE_SIZE = 4242