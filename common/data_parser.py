import pandas as pd
from scipy.io import arff
import glob

# required for importin modules from other directories
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from config import *

        
# parse the company dataset
# returns a list of 5 pandas dataframes if no year is given
# if 1,2,3,4 or 5 is passed as year the respective dataframe if said year is returned
def parse_companies(year=None):
    search_path = COMPANY_DATA_DIR + "\\*.*"
    file_list = glob.glob(search_path)
    assert len(file_list) == 5  # check if parsing worked

    if year == None:
        return [parse_companies(year) for year in range(MIN_COMPANY_YEAR, MAX_COMPANY_YEAR+1)]

    else:
        assert (year >= MIN_COMPANY_YEAR) and (year <= MAX_COMPANY_YEAR)
        data = arff.loadarff(file_list[year-1])
        return pd.DataFrame(data[0])
    


if __name__ == "__main__":
    df = parse_companies()
    print()