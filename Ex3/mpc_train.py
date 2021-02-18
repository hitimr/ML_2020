"""mpc_train.py

python mpc_train.py --num_participants 2 --log n --model_file ./models/mpc_mnist_relu.pt
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
parser.add_argument('--n_epochs',
                    type=int,
                    default=5,
                    metavar='C',
                    help='input n_epochs (default: mse)')
                    
parser.add_argument('--log',
                    type=str,
                    default='n',
                    metavar='C',
                    help='enable logging to folder log (default: n = disabled)')
parser.add_argument('--model_file',
                    type=str,
                    default='',
                    metavar='M',
                    help='file to save model (default: None)')
cl_args = parser.parse_args()

# initialize lists to monitor test loss and accuracy
# NUM_CLASSES = 10
num_participants = cl_args.num_participants
participants = POSSIBLE_PARTICIPANTS[:num_participants]


assert len(participants) == num_participants  # checking for shenanigans

# Instantiate and load the model
model = Net()


model_file_input = cl_args.model_file
if model_file_input:
    model_file_name = pathlib.Path(model_file_input)#model_file_input

n_epochs_input = cl_args.n_epochs
if n_epochs_input:
    n_epochs = n_epochs_input

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
        "per_epoch": [],
        "inference": {
            "total": 0,
            "per_batch": [],
            "per_image": [],
            "average_per_image": 0
        },
        "mem_before": mem_before,
        "mem_after": None
    }
    LOG_STR = ""
    runtime = 0
    predictions = []
    targets = []
    class_correct = [0] * NUM_CLASSES
    class_total = [0] * NUM_CLASSES
    valid_loss_min = +np.inf

    # Load model
    dummy_image = torch.empty([1, NUM_CHANNELS, IMG_WIDTH,
                               IMG_HEIGHT])  # is that the right way around? :D
    #model = crypten.load(model_file_name, dummy_model=Net(), src=0)
    model_mpc = crypten.nn.from_pytorch(model, dummy_image)
    model_mpc.encrypt(src=0)

    if pid == 0:
        print("Gonna train now...")

    #model_mpc.eval()  # prep model for evaluation

    before_test.wait()

    for epoch in range(1, n_epochs+1):
        # monitor losses
        train_loss = 0
        valid_loss = 0
        start = time()
        
        ###################
        # train the model #
        ###################
        iters = 0
        number_of_batches = len(train_loader)
        idx_to_show = np.arange(1, number_of_batches+1, int(number_of_batches/100))
        for batch_idx, (data, target) in enumerate(train_loader):
            if pid == 0 and batch_idx in idx_to_show:
                print(f"Batch: {(batch_idx+1) / (number_of_batches)*100:.2f}% --- {batch_idx+1}/{number_of_batches}")
            start_iter = time()
            data_enc = []
            label_eye = torch.eye(10)
            target = label_eye[target]
            if ws > 2:
                for idx, batch in enumerate(
                        split_data_even(data, ws - 1, data.shape[0])):
                    data_enc.append(crypten.cryptensor(batch, src=idx + 1))
                #data_enc = crypten.cat(data_enc, dim=0)
            else:
                data_enc.append(crypten.cryptensor(data, src=1))
            
            for tensor in data_enc:
                tensor.set_grad_enabled = True

            target_enc = crypten.cryptensor(target)
            #target_enc.set_grad_enabled = True

            model_mpc.train()  # prep model for evaluation
            
            # forward pass: compute predicted outputs by passing inputs to the model
            output = []
            start_batch_inference = time()
            # In each batch, each participant except the model holder has an equal share of the batch
            # Iterate over each participants share
            for dat in data_enc:
                output.append(model_mpc(dat))
            stop_batch_inference = time()
            output = crypten.cat(output, dim=0)
            #output.set_grad_enabled = True
            # convert output probabilities to predicted class
            pred = output.argmax(dim=1, one_hot=False)

            # calculate the loss
            if pid == 0:
                if output.shape != target_enc.shape:
                    print((output.shape, target_enc.shape))
            # loss = criterion(output, label) # pt
            loss = criterion(output, target_enc) #.get_plain_text()
        
            # clear the gradients of all optimized variables
            model.zero_grad()
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            #optimizer.step()
            model_mpc.update_parameters(learning_rate)
            # update running training loss
            train_loss += loss.get_plain_text().item() * data.size(0)

            # ### compare predictions to true label
            # # decrypt predictions
            # pred = pred.get_plain_text()
            # correct = np.squeeze(pred.eq(target.data.view_as(pred)))
            # # calculate test accuracy for each object class
            # predictions.append(pred)
            # targets.append(target)

                
            results["per_iter"].append(time() - start_iter)
            results["inference"]["per_batch"].append(stop_batch_inference - start_batch_inference)

            iters += 1
            iter_sync.wait()

        ###################
        # Save runtimes   #
        ###################
        stop = time()
        runtime = stop - start
        results["per_epoch"].append(runtime)
        results["total"] += runtime
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

        ######################    
        # validate the model #
        ######################
        model.eval()  # prep model for evaluation
        for data, label in valid_loader:
            data_enc = []
            if ws > 2:
                for idx, batch in enumerate(
                        split_data_even(data, ws - 1, data.shape[0])):
                    data_enc.append(crypten.cryptensor(batch, src=idx + 1))
                #data_enc = crypten.cat(data_enc, dim=0)
            else:
                data_enc.append(crypten.cryptensor(data, src=1))


            label_eye = torch.eye(10)
            label = label_eye[label]
            label_enc = crypten.cryptensor(label, src=0)
            # forward pass: compute predicted outputs by passing inputs to the model
            output = [model_mpc(dat) for dat in data_enc]
            output = crypten.cat(output, dim=0)
            if pid == 0:
                if output.shape != label_enc.shape:
                    print((output.shape, label_enc.shape))
            # calculate the loss
            loss = criterion(output, label_enc).get_plain_text()
            # update running validation loss 
            valid_loss = loss.item() * data.size(0)

        # print training/validation statistics 
        # calculate average loss over an epoch
        train_loss = train_loss / len(train_loader.sampler)
        valid_loss = valid_loss / len(valid_loader.sampler)

        tmp_str = f"Epoch: {epoch} \tTraining Loss: {train_loss:.6f} \tValidation Loss: {valid_loss:.6f}\n"
        LOG_STR += tmp_str
        if pid == 0:
            print(tmp_str)

        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            model_dec = model_mpc.decrypt()

            tmp_str = f"Validation loss decreased ({valid_loss_min:.6f} --> {valid_loss:.6f}).  Saving model ...\n"
            LOG_STR += tmp_str
            if pid == 0:
                print(tmp_str)
                print(f"Saving model at {model_file_name}")
                torch.save(model_mpc.state_dict(), model_file_name)
                crypten.save(model_mpc.state_dict(), model_file_name)
            valid_loss_min = valid_loss
            model_mpc.encrypt()

    if pid == 0:
        print("Done evaluating...")

    after_test.wait()

    if pid == 0:
        print("Ouputing information...")

    # calculate and print avg test loss
    #test_loss = test_loss / len(test_loader.sampler)
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
    # LOG_STR = f"Rank: {pid}\nWorld_Size: {ws}\n\n"
    # LOG_STR += f"Test runtime: {runtime:5.2f}s\n"
    # LOG_STR += f"Test Loss: {test_loss:.6}\n"
    # LOG_STR += "\n"
    # for i in range(NUM_CLASSES):
    #     if class_total[i] > 0:
    #         LOG_STR += f"Test Accuracy of {i:5}: " \
    #               f"{100 * class_correct[i] / class_total[i]:3.0f}% " \
    #               f"({np.sum(class_correct[i]):4} / {np.sum(class_total[i]):4} )"
    #         LOG_STR += "\n"
    #     else:
    #         LOG_STR += f"Test Accuracy of {classes[i]}: N/A (no training examples)"
    #         LOG_STR += "\n"
    # LOG_STR += f"\nTest Accuracy (Overall): {100. * np.sum(class_correct) / np.sum(class_total):3.0f}% " + \
    #       f"( {np.sum(class_correct)} / {np.sum(class_total)} )"
    
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
