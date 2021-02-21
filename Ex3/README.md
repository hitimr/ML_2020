# Usage

## Requirements

python >= 3.7

is the official requirement listed on <https://pypi.org/project/crypten/>.

## Packages

Crypten0.1 (latest release) requires python >= 3.7 and torch==1.4.0.

It also seems to install just fine under python3.8, although we have run into issues running some of our code involving crypten there - so we recommend python3.7.9!

Installation can be done via using the provided requirements.txt or Pipfile.

### requirements.txt

numpy==1.20.0
pandas
matplotlib
seaborn
jupyterlab
mpyc==0.7
crypten==0.1
torch==1.4.0
tqdm
psutil
ipywidgets

### Pipfile

numpy = "1.20.0"
pandas = "*"
matplotlib = "*"
seaborn = "*"
jupyterlab = "*"
scikit-learn = "0.24.1"
mpyc = "0.7"
crypten = "0.1"
torch = "1.4.0"
tqdm = "*"
psutil = "*"
ipywidgets = "*"

## Project structure

CNN/
ex3_lib/
log/
models/
mpc/

common.py
config.py
model_creator_v1.1.ipynb
mpc_evaluate.py
mpc_train.py
plots.ipynb
tensor_benchmark.ipynb

## Example commands to start scripts
