# Usage

## Requirements

python >= 3.7

is the official requirement listed on <https://pypi.org/project/crypten/>.

## Packages

Crypten0.1 (latest release) requires python >= 3.7 and torch==1.4.0.

It also seems to install just fine under python3.8, although we have run into issues running some of our code involving crypten there - so we recommend python3.7.9!

Installation can be done via using the provided requirements.txt or Pipfile.

Pip:

```bash
pip install -r requirements.txt
```

Pipenv:

```bash
pipenv install
```

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

data/       # Created if not present and handled by dataloader (downloads data if missing)
            # If issues arise, please delete this folder retry
CNN/        # Training and evaluation of CNN model
ex3_lib/    # Convenience functions for data and dir management
log/        # log containing the collected data from benchmarks
models/     # contains models, dataset/model configs
mpc/        # More convenience functions for data setup and profiling

common.py                    # Contains common functions used across
config.py                    # Contains project level configs, such as paths
**model_creator_v1.1.ipynb** # Notebook used for in the clear training and evaluation of models
**mpc_evaluate.py**          # Script for MPC evaluation/prediction benchmark
**mpc_train.py**             # Script for MPC training benchmark
(plots.ipynb)                # Plots...
**tensor_benchmark.ipynb**   # Notebook for benchmarking Torch and CrypTen tensors

## Example commands to start scripts

### Evaluation (Prediction performance)

Benchmarks the prediction/inference in a secure MPC setting.

Example call to start with 2 participants/processes (rank 0 always holds the model, while the additional parties split the data evenly).

```bash
python mpc_evaluate.py --num_participants 2 --dataset fashion 
```

### Training

Benchmarks the prediction/inference in a secure MPC setting.

Example call to start training with 2 participants/processes (rank 0 always holds the model, while the additional parties split the data evenly) on the Fashion MNIST dataset using only 5000 of the 60000 samples and 2 epochs.

```bash
python mpc_train.py --num_participants 2 --dataset fashion --samples 5000 --epochs 2
```

### In the clear training and evaluation

Use the jupyter notebook called `model_creator_v1.1.ipynb`.

### Torch vs CrypTen Tensors

Use the jupyter notebook called `tensor_benchmarks.ipynb`.
