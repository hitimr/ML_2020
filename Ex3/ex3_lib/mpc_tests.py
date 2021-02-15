"""mpc_tests.py
Some test functions I wrote to test MPC execution."""
import torch
import crypten
import crypten.communicator as mpc_comm # the communicator is similar to the MPI communicator for example
from crypten import mpc
from .mpc_helpers import *

world_size = 5
        
@mpc_check
@mpc.run_multiprocess(world_size=world_size)
def test():
    ws = mpc_comm.get().world_size
    rank = mpc_comm.get().get_rank()
    print(rank)
    tens = torch.tensor([x for x in range(ws+1)])
    #for rank in range(ws):
    crypten.save(tens, f"test_{rank}.pth", src=rank)
        #crypten.save_from_party(tens, f"test_{rank}", src=rank)

@mpc_check
@mpc.run_multiprocess(world_size=world_size)
def test_load():
    ws = mpc_comm.get().world_size
    rank = mpc_comm.get().get_rank()
    print(rank)
    data = []
    for rank in range(ws):
        data.append(crypten.load(f"test_{rank}.pth", src=rank))
    print(data[0])
    print(data[0].get_plain_text())
        
def test_solo(world_size=world_size):
    tens = torch.tensor([x for x in range(world_size+1)])
    for rank in range(world_size):
        crypten.save(tens, f"test_{rank}.pth", src=rank)

if __name__=="__main__":
    test()
    test_load()
    #test_solo()   
