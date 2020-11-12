import pandas as pd
from scipy.io import arff
import glob

# required for importin modules from other directories
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from config import *

 

def parse_companies(year=None):
    """parse the company dataset

    Returns: dataframe or [dataframe]: returns a list of 5 pandas dataframes if
        no year is given if 1,2,3,4 or 5 is passed as year the respective
        dataframe if said year is returned

    """    
    search_path = COMPANY_DATA_DIR + "/*.*"
    file_list = glob.glob(search_path)
    assert len(file_list) == 5  # check if parsing worked

    if year == None:
        return [parse_companies(year) for year in range(MIN_COMPANY_YEAR, MAX_COMPANY_YEAR+1)]

    else:
        assert (year >= MIN_COMPANY_YEAR) and (year <= MAX_COMPANY_YEAR)
        data = arff.loadarff(file_list[year-1])
        return pd.DataFrame(data[0])



def parse_breastCancer(data_set):
    
    """imports the breast canser dataset. depending on the data_set. learn-
    (lrn), solution- (sol) or test-set (tes) are returned

    Raises: ValueError: invalid argument

    Returns: dataframe: pandas dataframe containing the requested dataset
    """    
    if data_set == "lrn":
        df = pd.read_csv(BREASTCANCER_DATA_DIR + "/breast-cancer-diagnostic.shuf.lrn.csv")
        assert df.size == BREASTCANCER_LRN_SIZE
        return df

    if data_set == "sol":
        df = pd.read_csv(BREASTCANCER_DATA_DIR + "/breast-cancer-diagnostic.shuf.sol.ex.csv")
        assert df.size == BREASTCANCER_SOL_SIZE
        return df
    
    if data_set == "tes":
        df = pd.read_csv(BREASTCANCER_DATA_DIR + "/breast-cancer-diagnostic.shuf.tes.csv")
        assert df.size == BREASTCANCER_TES_SIZE
        return df

    raise ValueError("Improper argument. Allowed arguments are lrn, sol, test")

def parse_amazon(data_set):
        
    """imports the amazon review dataset. depending on the data_set. learn-
    (lrn), solution- (sol) or test-set (tes) are returned

    Raises: ValueError: invalid argument

    Returns: dataframe: pandas dataframe containing the requested dataset
    """    
    if data_set == "lrn":
        df = pd.read_csv(AMAZON_DATA_DIR + "/amazon_review_ID.shuf.lrn.csv")
        assert df.size == AMAZON_LRN_SIZE
        return df

    if data_set == "sol":
        df = pd.read_csv(AMAZON_DATA_DIR + "/amazon_review_ID.shuf.sol.ex.csv")
        assert df.size == AMAZON_SOL_SIZE
        return df
    
    if data_set == "tes":
        df = pd.read_csv(AMAZON_DATA_DIR + "/amazon_review_ID.shuf.tes.csv")
        assert df.size == AMAZON_TES_SIZE
        return df

    raise ValueError("Improper argument. Allowed arguments are lrn, sol, test")


def parse_congressional_voting(data_set):        
    """imports the congressional voting dataset. depending on the data_set. train-
    (train), test- (test) or sampleSubmission-set (sampleSubmission) are returned

    Raises: ValueError: invalid argument

    Returns: dataframe: pandas dataframe containing the requested dataset
    """    
    if data_set == "sampleSubmission":
        df = pd.read_csv(CONGRESSIONAL_VOTING_DIR + "/CongressionalVotingID.shuf.sol.ex.csv")
        assert df.size == CONGRESSIONALVOTING_SAMPLE_SIZE
        return df

    if data_set == "test":
        df = pd.read_csv(CONGRESSIONAL_VOTING_DIR + "/CongressionalVotingID.shuf.tes.csv")
        assert df.size == CONGRESSIONALVOTING_TEST_SIZE
        return df
    
    if data_set == "train":
        df = pd.read_csv(CONGRESSIONAL_VOTING_DIR + "/CongressionalVotingID.shuf.lrn.csv")
        assert df.size == CONGRESSIONALVOTING_TRAIN_SIZE
        return df

    raise ValueError("Improper argument. Allowed arguments are lrn, sol, test")

def parse_heart_disease(data_set = "sma"):        
    """imports the heart disease dataset.

    Raises: ValueError: invalid argument

    Returns: dataframe: pandas dataframe containing the requested dataset
    """
    if data_set == "sma":
        df = pd.read_csv(HEART_DISEASE_DIR + "/heart.csv")
        assert df.size == HEART_DISEASE_SIZE
        return df
    elif data_set == "big":
        df = pd.read_csv(HEART_DISEASE_DIR + "/HEART_DISEASE.csv")
        assert df.size == HEART_DISEASE_SIZE
        return df

    raise ValueError("Improper argument. Allowed arguments are sma, big")


if __name__ == "__main__":
    df = parse_congressional_voting("train")
    print(df.size)