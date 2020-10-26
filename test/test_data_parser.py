import pytest

# required for importin modules from other directories
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

from config import *
from common.data_parser import *



def test_parse_companies():
    df_list1 = []
    for i in range(MIN_COMPANY_YEAR, MAX_COMPANY_YEAR+1):
        df_list1.append(parse_companies(i))

    df_list2 = parse_companies()

    assert len(df_list1) == 5
    assert len(df_list2) == 5

    for i in range(len(df_list1)):
        assert df_list1[i].size == df_list2[i].size


def test_parse_breastCancer():
    parse_breastCancer("lrn")
    parse_breastCancer("sol")
    parse_breastCancer("tes")

    with pytest.raises(Exception):
        parse_breastCancer()

    with pytest.raises(Exception):
        parse_breastCancer("Gay")


def test_parse_amazon():
    parse_amazon("lrn")
    parse_amazon("sol")
    parse_amazon("tes")

    with pytest.raises(Exception):
        parse_amazon()

    with pytest.raises(Exception):
        parse_amazon("42")


def test_parse_congressional_voting():
    parse_congressional_voting("sampleSubmission")
    parse_congressional_voting("test")
    parse_congressional_voting("train")

    with pytest.raises(Exception):
        parse_congressional_voting()

    with pytest.raises(Exception):
        parse_congressional_voting("Anal")

if __name__ == "__main__":
    test_parse_companies()



