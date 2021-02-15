"""dir_setup.py

Functions for data directory setup and generation of filename dictionary.
"""
import pathlib
import shutil

def rm_files_in_dir(directoy:pathlib.Path, recursive=False):
    if not directory.exists():
        print(f"{directory} does not exist")
    else:
        if recursive:
            # Simple non-recursive glob --> doesn't look into subdirectories
            for file in [x for x in directory.glob("*") if x.is_file()]:
                file.unlink()
            print(f"{directory} cleaned!")
        else:
            # Use recursive glob
            for file in [x for x in directory.rglob("*") if x.is_file()]:
                file.unlink()
            print(f"{directory} and it's subdirectories cleaned!")

def check_and_mkdir(directory:pathlib.Path):
    if not directory.exists():
        print(f"Directory {directory} created.")
        directory.mkdir()

def rm_dir(directory:pathlib.Path):
    shutil.rmtree(directory)
    
def get_filenames(directory:pathlib.Path, participants):
    # Specify file locations to save each piece of data
    filenames = {
        "features": directory / "features.pth",
        "labels": directory / "labels.pth",
        "b_true": directory / "b_true.pth",
        "test_features": directory / "test_features.pth",
        "targets": directory / "targets.pth",
        "w_true": directory / "w_true.pth",
    }

    rank = 0
    for u in participants:
        filenames["labels_"+u] = (directory / (f"labels_{u}.pth"), rank)
        filenames["features_"+u] = (directory / (f"features_{u}.pth"), rank)
        rank += 1
    return filenames

def setup(participants, tmp_dir_name="./TMP"):
    num_participants = len(participants)
    TMP_DIR = pathlib.Path(tmp_dir_name)
    print(f"Our temporary data will land here: {TMP_DIR}")
    check_and_mkdir(TMP_DIR)
    filenames = get_filenames(TMP_DIR, participants)
    return TMP_DIR, filenames, num_participants 

POSSIBLE_PARTICIPANTS = ("alice, bob, clara, daniel, " + 
    "elina, franz, georg, hilda, ilya, julia, karin, luke, " +
    "martin, nadia, olaf, peter, queenie, rasmus, sarah, tal, " +
    "ulyana, valerie, walter, xander, ymir, zorro").split(", ")
