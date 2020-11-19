import pathlib
import os

def get_data_path(folder:str, filename:str):
    ''' Helper function to retrieve data.

    args: 
        folder... name of folder/dataset
        filename... name of file to get
    '''
    cwd = pathlib.Path(os.getcwd())
    data_dir = cwd.parent / "Datasets"
    return data_dir / folder / filename