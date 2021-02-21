"""mpc_train.py

python mpc_train.py --num_participants 2 --model_file ./models/mpc_mnist_relu_ts0.5perc.pt 

python mpc_train.py --num_participants 2 --dataset fashion
"""

import argparse
import pathlib
import sys
import numpy as np

import crypten
import crypten.communicator as comm  # the communicator is similar to the MPI communicator for example
# Check wether the correct python version is installed and init crypten
assert sys.version_info[0] == 3 and sys.version_info[
    1] >= 7, "python >3.7 is required!"
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

from mpc.mpc_profile import * 

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
parser.add_argument('--model_file',
                    type=str,
                    default='',
                    metavar='M',
                    help='file to save model (default: None)')
parser.add_argument('--dataset',
                    type=str,
                    default='mnist',
                    metavar='D',
                    help='dataset to train on (default: mnist)')
parser.add_argument('--epochs',
                    type=int,
                    default=5,
                    metavar='E',
                    help='Number of epochs to train for (default: 5)')
parser.add_argument('--samples',
                    type=int,
                    default=0,
                    metavar='S',
                    help='Number of samples to use for training (default: 0 = all), range = [0, 60000]')
cl_args = parser.parse_args()

epochs_input = cl_args.epochs
samples_input = cl_args.samples

dataset_input = cl_args.dataset
if dataset_input=="mnist":
    print("Evaluating with MNIST dataset")
    from models.mnist_relu_conf import *
    from models.mnist_relu_conf import ReLUMLP as Net

    sample_size = samples_input
    n_epochs = epochs_input

    from mpc.setup_mnist import setup_data
    setup_data()
    print(model_file_name)

elif dataset_input=="fashion":
    print("Evaluating with FASHION dataset")
    from models.fashion_relu_conf import *
    from models.fashion_relu_conf import ReLUMLP as Net

    sample_size = samples_input
    n_epochs = epochs_input

    from mpc.setup_fashion import setup_data
    setup_data()
    print(model_file_name)

else:
    raise ValueError("Invalid dataset choice!")

# dataset_input = cl_args.dataset
# if dataset_input=="mnist":
#     print("Training with MNIST dataset")
#     from models.mnist_relu_conf import ReLUMLP as Net
#     from models.mnist_relu_conf import *
#     from mpc.setup_mnist import *
# elif dataset_input=="fashion":
#     print("Training with FASHION dataset")
#     from models.fashion_relu_conf import ReLUMLP as Net
#     # from models.fashion_relu_conf import *
#     from mpc.setup_fashion import *
# else:
#     raise ValueError("Invalid dataset choice!")

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

# Logging functionality
log_dir = pathlib.Path("./log/train")
check_and_mkdir(log_dir)
memory_dir= log_dir / "memory"
runtimes_dir = log_dir / "runtimes"
results_dir = log_dir / "results"
check_and_mkdir(memory_dir) 
check_and_mkdir(runtimes_dir)
check_and_mkdir(results_dir)

print(f"Logging to {str(log_dir.absolute())}")
torch.set_num_threads(1)  # One thread per participant


@mpc.run_multiprocess(world_size=num_participants)
def train_model_mpc(): 
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

    # Setup log file per process
    postfix = f"{DATASET_NAME}_{ws}p_{pid}.log"
    memory_log = memory_dir / postfix
    runtimes_log = runtimes_dir / postfix
    results_log = results_dir / postfix

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
            # pred = output.argmax(dim=1, one_hot=False)

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
                #orch.save(model_mpc.state_dict(), model_file_name)
                torch.save(model_dec, model_file_name)
            valid_loss_min = valid_loss
            model_mpc.encrypt(src=0)
        log_memory(memory_log)

    if pid == 0:
        print("Done training...")

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

    with open(f"./log/train/stdout_{pid}", "w") as f:
        f.write(LOG_STR)
    
    done.wait()
    mem_after = get_process_memory()
    results["mem_after"] = mem_after
    with open(results_log, 'w') as f:
        f.write(str(results))
    if pid == 0:
        with open(results_dir / f'latest_{pid}.txt', 'w') as f:
            f.write(str(results))

    return results
#### Main train function

if __name__ == "__main__":
    TOTAL_TIME = time()

    RESULTS = train_model_mpc()

    TOTAL_TIME = elapsed_since(TOTAL_TIME)
    mem_before = RESULTS[0]["mem_before"]
    mem_after = RESULTS[0]["mem_after"]
    print(f"Memory usage: memory before: {str(mem_before):}, after: {str(mem_after):}, consumed: {str((mem_after[0] - mem_before[0], mem_after[1])):}; exec time: {str(TOTAL_TIME)}")

    print(RESULTS[0])


# results = {
#         "total": 0,
#         "per_iter": [],
#         "per_epoch": [],
#         "inference": {
#             "total": 0,
#             "per_batch": [],
#             "per_image": [],
#             "average_per_image": 0
#         },
#         "mem_before": mem_before,
#         "mem_after": None
#     }