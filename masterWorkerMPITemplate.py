################################################################################
# Master-worker async polling set-up
################################################################################

import torch
import sys
import os
#sys.path.append("../")
from distributed import init_workers
import torch.distributed as dist
import threading
import time
import numpy as np

################################################################################
# Print function useful for debugging async parallel programs
################################################################################
def print_some(thing):
    original_stdout = sys.stdout  # Save a reference to the original standard output
    with open('print_output.txt', 'a') as f:
        sys.stdout = f  # Change the standard output to the file we created.
        print(thing)
        sys.stdout = original_stdout  # Reset the standard output to its original value

################################################################################
# Master code
################################################################################
def master_setup():
    # initial setup of recv from everyone
    reqs = []
    grads = []
    for i in range(1, world_size):
        grads.append(torch.zeros(2, 2))
        #reqs.append(irecv(grads[-1], i))
        reqs.append(dist.irecv(grads[-1], i))
        
    return reqs,grads

def master_loop(reqs,grads,CONTINUE,x):
    # loop to check for an irecv completed
    step = 0.1
    MAX_UPDATES = 100
    updates = 0
    i = 0
    grad_order = []
    while True:
        if reqs[i].is_completed():
            x -= step * grads[i]
            updates += 1
            grad_order.append(i+1)
            if updates >= MAX_UPDATES:
                # send message to others to exit...
                break
            #send(CONTINUE, i + 1)
            dist.send(CONTINUE, i + 1)
            #send(x, i + 1)
            dist.send(x, i + 1)

            #reqs[i] = irecv(grads[i], i + 1)
            reqs[i] = dist.irecv(grads[i], i+1)
            
        i = (i + 1) % (world_size - 1)

    print(f"master loop exiting,")
    print(f"gradient order: {grad_order}")

def master_cleanup(reqs,FINISH):
    # clean-up (tell other processes to shut down)
    i = 0
    finishes_sent = 0
    not_sent = [1 for i in range(world_size-1)]
    while True:
        if (reqs[i].is_completed()) and not_sent[i]:
            not_sent[i] = 0
            #send(FINISH, i + 1)
            dist.send(FINISH, i + 1)
            finishes_sent += 1
            if finishes_sent >= world_size - 1:
                break
        i = (i + 1) % (world_size - 1)

def master():

    x = torch.tensor([[1., -1.], [1., -1.]])
    CONTINUE = torch.tensor([1])
    FINISH = torch.tensor([0])

    # initial send to everyone
    dist.broadcast(x,0)

    reqs,grads = master_setup()

    print_some(reqs)
    master_loop(reqs,grads,CONTINUE,x)
    print_some(reqs)


    master_cleanup(reqs,FINISH)

    print(f"master exiting with x = {x}")

################################################################################
# Worker code
################################################################################
def worker_loop(x,continue_flag):
    sigma = 0.1
    ngrads = 0
    while True:
        grad = 2*x + sigma*torch.randn(2,2)
        #time.sleep(0.1*np.random.uniform())
        #time.sleep(0.1*global_rank)
        #req = isend(grad, 0)
        req = dist.isend(grad, 0)
        print_some(f"worker {global_rank} waiting...")
        req.wait()
        ngrads += 1
        #recv(continue_flag, 0)
        dist.recv(continue_flag, 0)
        if continue_flag[0] == 0:
            break
        #recv(x, 0)
        dist.recv(x, 0)
    print(f"worker {global_rank} exiting after computing {ngrads} gradients")

def worker():


    x = torch.zeros(2,2)
    continue_flag = torch.tensor([-1])
    # initial send to everyone
    dist.broadcast(x, 0)
    worker_loop(x,continue_flag)

    #print(f"worker {global_rank} exiting with x = {x}")

################################################################################
# Distributed set-up
################################################################################
distributed_backend = 'mpi'

global_rank, world_size = init_workers(distributed_backend)

print(f"my global rank is {global_rank}")
if global_rank==0:
    print(f"world size: {world_size}")
    if os.path.exists("print_output.txt"):
        os.remove("print_output.txt")

################################################################################
# Lasso set-up
################################################################################
m = 1000
d = 2000
if global_rank==0:
    A = torch.load('A.torch')
    b = torch.rand((m,1))
else:
    A = torch.empty((m,d))
    b = torch.empty((m,1))

dist.broadcast(A,0)
dist.broadcast(b,0)

################################################################################
# Main
################################################################################

if global_rank==0:
    master()
else:
    worker()


