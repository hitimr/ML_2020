# required for importin modules from other directories
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import pytest
from common.DataParser import *

def test_parse_test_housePrices():
    df = parse_test_housePrices()
    assert df.empty == False
    df = parse_test_housePrices(splitData=False)
    assert df.empty == False

    x,y = parse_test_housePrices(splitData=True)
    assert x.empty == False
    assert y.empty == False
    print(x)
    print(y)


if __name__ == "__main__":
    test_parse_test_housePrices()