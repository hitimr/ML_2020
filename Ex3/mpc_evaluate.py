"""evaluate a model via MPC

Example call to start with 6 participants/processes and pipe output into log file.

python mpc_evaluate.py --num_participants 6 --log n > log/6p.log
"""
import argparse
import pathlib
import sys
import numpy as np

import crypten
import crypten.communicator as comm  # the communicator is similar to the MPI communicator for example
# Check wether the correct python version is installed and init crypten
assert sys.version_info[0] == 3 and sys.version_info[
    1] == 7, "python 3.7 is required!"
print(f"Okay, good! You have: {sys.version_info[:3]}")
# Now we can init crypten!
crypten.init()
###
from crypten import mpc
from multiprocessing import Barrier
from tqdm import tqdm
from time import time

# Functions for setting up the directory structure
from ex3_lib.data import split_data_even
from ex3_lib.dir_setup import POSSIBLE_PARTICIPANTS, check_and_mkdir

from models.mnist_relu_conf import ReLUMLP as Net
from models.mnist_relu_conf import *
from mpc.setup_mnist import *
from mpc.profile import * 

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--num_participants',
                    type=int,
                    default=2,
                    metavar='P',
                    help='input number of participants (default: 2)')
parser.add_argument('--criterion',
                    type=str,
                    default='mse',
                    metavar='C',
                    help='input loss criterion to use (default: mse)')
parser.add_argument('--log',
                    type=str,
                    default='n',
                    metavar='C',
                    help='enable logging to folder log (default: n = disabled)')
cl_args = parser.parse_args()

# initialize lists to monitor test loss and accuracy
# NUM_CLASSES = 10
num_participants = cl_args.num_participants
participants = POSSIBLE_PARTICIPANTS[:num_participants]


assert len(participants) == num_participants  # checking for shenanigans

# Instantiate and load the model
model = Net()
print(f"Loading model from {model_file_name}")
model.load_state_dict(torch.load(model_file_name))

#convert_legacy_config() # LEGACY
#model_mpc = crypten.nn.from_pytorch(model, dummy_image)

# Barriers for synchronisation
before_test = Barrier(num_participants)
after_test = Barrier(num_participants)
done = Barrier(num_participants)

iter_sync = Barrier(num_participants)

### Choose a loss criterion
# Crypten loss criteria
criterion_type = cl_args.criterion
if criterion_type == "mse":
    criterion = crypten.nn.MSELoss()
elif criterion_type == "crossentropy":
    criterion = crypten.nn.CrossEntropyLoss()
else:
    raise ValueError(f"{criterion_type} is not a supported criterion! See help for supported types")
log_switch = cl_args.log
if log_switch == "y":
    check_and_mkdir(pathlib.Path("./log")) # log each processes results to file
else:
    print("Logging to file disabled")

torch.set_num_threads(1)  # One thread per participant
TOTAL_TIME = time()
@mpc.run_multiprocess(world_size=num_participants)
def test_model_mpc():
    mem_before = get_process_memory()   
    runtime = 0
    pid = comm.get().get_rank()
    ws = comm.get().world_size
    name = participants[pid]
    if pid == 0:
        print(f"Hello from the main process (rank#{pid} of {ws})!")
        print(f"My name is {name}.")
        print(f"My colleagues today are: ")
        print(participants)
    results = {
        "total": 0,
        "per_iter": [],
        "inference": {
            "total": 0,
            "per_batch": [],
            "per_image": [],
            "average_per_image": 0
        },
        "mem_before": mem_before,
        "mem_after": None
    }
    predictions = []
    targets = []
    class_correct = [0] * NUM_CLASSES
    class_total = [0] * NUM_CLASSES

    # Load model
    dummy_image = torch.empty([1, NUM_CHANNELS, IMG_WIDTH,
                               IMG_HEIGHT])  # is that the right way around? :D
    #model = crypten.load(model_file_name, dummy_model=Net(), src=0)
    model_mpc = crypten.nn.from_pytorch(model, dummy_image)
    model_mpc.encrypt(src=0)

    if pid == 0:
        print("Gonna evaluate now...")

    test_loss = 0.0
    model_mpc.eval()  # prep model for evaluation

    before_test.wait()
    start = time()
    
    iters = 0
    for data, target in tqdm(test_loader, position=0):  #, desc=f"{name}"):
        start_iter = time()
        data_enc = []
        if ws > 2:
            for idx, batch in enumerate(
                    split_data_even(data, ws - 1, data.shape[0])):
                data_enc.append(crypten.cryptensor(batch, src=idx + 1))
            #data_enc = crypten.cat(data_enc, dim=0)
        else:
            data_enc.append(crypten.cryptensor(data, src=1))

        target_enc = crypten.cryptensor(target, src=0)

        # forward pass: compute predicted outputs by passing inputs to the model
        output = []
        start_batch_inference = time()
        # In each batch, each participant except the model holder has an equal share of the batch
        # Iterate over each participants share
        for dat in data_enc:
            output.append(model_mpc(dat))
        stop_batch_inference = time()

        output = crypten.cat(output, dim=0)
        # convert output probabilities to predicted class
        pred = output.argmax(dim=1, one_hot=False)
        # calculate the loss
        if pid == 0:
            if pred.shape != target_enc.shape:
                print((pred.shape, target_enc.shape))
        loss = criterion(pred, target_enc).get_plain_text()
        # update test loss
        test_loss += loss.item() * data.size(0)

        ### compare predictions to true label
        # decrypt predictions
        pred = pred.get_plain_text()
        correct = np.squeeze(pred.eq(target.data.view_as(pred)))
        # calculate test accuracy for each object class
        predictions.append(pred)
        targets.append(target)
        for i in range(len(target)):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1
        results["per_iter"].append(time() - start_iter)
        results["inference"]["per_batch"].append(stop_batch_inference - start_batch_inference)

        iters += 1
        iter_sync.wait()

    stop = time()
    runtime = stop - start
    results["total"] = runtime
    results["average_per_iter"] = np.mean(results["per_iter"])
    results["inference"]["total"] = np.sum(results["inference"]["per_batch"])
    results["inference"]["per_image"] = [x/batch_size for x in results["inference"]["per_batch"]]
    results["inference"]["average_per_image"] = np.mean(results["inference"]["per_image"])
    # results = {
    #     "total": 0,
    #     "per_iter": [],
    #     "inference": {
    #         "total": 0,
    #         "per_batch": [],
    #         "per_image": [],
    #         "average_per_image": 0
    #     }
    # }

    if pid == 0:
        print("Done evaluating...")

    after_test.wait()

    if pid == 0:
        print("Ouputing information...")

    # calculate and print avg test loss
    test_loss = test_loss / len(test_loader.sampler)
    # if pid == 0:
    #     print(f"Test runtime: {runtime:5.2f}s\n\n")
    #     print(f"Test Loss: {test_loss:.6}\n")
    #     # Print accuracy per class
    #     for i in range(NUM_CLASSES):
    #         if class_total[i] > 0:
    #             print(
    #                 f"Test Accuracy of {i:5}: "
    #                 f"{100 * class_correct[i] / class_total[i]:3.0f}% "
    #                 f"({np.sum(class_correct[i]):4} / {np.sum(class_total[i]):4} )"
    #             )
    #         else:
    #             print(
    #                 f"Test Accuracy of {classes[i]}: N/A (no training examples)"
    #             )
    #     # Print overall accuracy
    #     print(
    #         f"\nTest Accuracy (Overall): {100. * np.sum(class_correct) / np.sum(class_total):3.0f}% "
    #         f"( {np.sum(class_correct)} / {np.sum(class_total)} )")

    # Gather log
    LOG_STR = f"Rank: {pid}\nWorld_Size: {ws}\n\n"
    LOG_STR += f"Test runtime: {runtime:5.2f}s\n"
    LOG_STR += f"Test Loss: {test_loss:.6}\n"
    LOG_STR += "\n"
    for i in range(NUM_CLASSES):
        if class_total[i] > 0:
            LOG_STR += f"Test Accuracy of {i:5}: " \
                  f"{100 * class_correct[i] / class_total[i]:3.0f}% " \
                  f"({np.sum(class_correct[i]):4} / {np.sum(class_total[i]):4} )"
            LOG_STR += "\n"
        else:
            LOG_STR += f"Test Accuracy of {classes[i]}: N/A (no training examples)"
            LOG_STR += "\n"
    LOG_STR += f"\nTest Accuracy (Overall): {100. * np.sum(class_correct) / np.sum(class_total):3.0f}% " + \
          f"( {np.sum(class_correct)} / {np.sum(class_total)} )"
    
    if pid == 0:
        print(LOG_STR)

    if log_switch=="y":
        with open(f"log/test_log_rank{pid}", "w") as f:
            f.write(LOG_STR)

    done.wait()
    mem_after = get_process_memory()
    results["mem_after"] = mem_after
    return results

TOTAL_TIME = elapsed_since(TOTAL_TIME)
RESULTS = test_model_mpc()
mem_before = RESULTS[0]["mem_before"]
mem_after = RESULTS[0]["mem_after"]
print(f"Memory usage: memory before: {mem_before:}, after: {mem_after:}, consumed: {mem_after - mem_before:}; exec time: {TOTAL_TIME}")
print(RESULTS[0])

# Test runtime: 761.02s

# Test Loss: 1.2081e+11

# Test Accuracy of     0:  98% ( 964 /  980 )
# Test Accuracy of     1:  98% (1110 / 1135 )
# Test Accuracy of     2:  90% ( 925 / 1032 )
# Test Accuracy of     3:  94% ( 949 / 1010 )
# Test Accuracy of     4:  90% ( 885 /  982 )
# Test Accuracy of     5:  86% ( 763 /  892 )
# Test Accuracy of     6:  93% ( 889 /  958 )
# Test Accuracy of     7:  94% ( 962 / 1028 )
# Test Accuracy of     8:  83% ( 806 /  974 )
# Test Accuracy of     9:  89% ( 894 / 1009 )

# Test Accuracy (Overall):  91% ( 9147 / 10000 )
