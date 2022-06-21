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
# Distributed set-up
################################################################################
distributed_backend = 'nccl'

global_rank, world_size = init_workers(distributed_backend)

print(f"my global rank is {global_rank}")
if global_rank==0:
    print(f"world size: {world_size}")
    if os.path.exists("print_output.txt"):
        os.remove("print_output.txt")

torch.cuda.set_device(global_rank) # set GPU to be the one with id = global_rank

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
# Setup send/recv for nccl work-around
################################################################################
if distributed_backend == 'nccl':
    groups = [None]
    for i in range(1,world_size):
        groups.append(dist.new_group([0, i]))

    # over-write comms
    def send_wrapper(tensor,node):
        dist.broadcast(tensor,0,groups[node])

    def recv_wrapper(tensor,receiving_node):
        dist.broadcast(tensor, 0, groups[receiving_node])


    #req = isend(grad, 0)
    def isend_wrapper(tensor,sending_node):
        return dist.broadcast(tensor,sending_node,groups[sending_node],async_op=True)

    #req = irecv(grads[-1], i)
    def irecv_wrapper(tensor,sending_node):
        return dist.broadcast(tensor, sending_node, groups[sending_node], async_op=True)


else:

    def send_wrapper(tensor, node):
        dist.send(tensor,node)

    def recv_wrapper(tensor,receiving_node):
        dist.recv(tensor,0)

    def isend_wrapper(tensor,sending_node):
        return dist.isend(tensor,0)


    def irecv_wrapper(tensor, sending_node):
        return dist.irecv(tensor,sending_node)




################################################################################
# Setup is_completed() patch for gloo bug
################################################################################
def daemon_thread(req):
    req.wait()
def check_a_req(req):
    if distributed_backend != "gloo":
        return req.is_completed()
    else:
        return (not req.is_alive())

def set_a_req(reqs,i):
    if distributed_backend=="gloo":
        t = threading.Thread(target=daemon_thread, args=(reqs[i],), daemon=True)
        t.start()
        reqs[i]=t

################################################################################
# Master code
################################################################################
def master_setup():
    # initial setup of recv from everyone
    reqs = []
    grads = []
    for i in range(1, world_size):
        grads.append(torch.zeros(2, 2).cuda())
        #reqs.append(irecv(grads[-1], i))
        reqs.append(irecv_wrapper(grads[-1], i))
        set_a_req(reqs, i - 1)  # used only for gloo backend
    return reqs,grads

def master_loop(reqs,grads,CONTINUE,x):
    # loop to check for an irecv completed
    step = 0.1
    MAX_UPDATES = 100
    updates = 0
    i = 0
    grad_order = []
    while True:
        if check_a_req(reqs[i]):
            x -= step * grads[i]
            updates += 1
            grad_order.append(i+1)
            if updates >= MAX_UPDATES:
                # send message to others to exit...
                break
            #send(CONTINUE, i + 1)
            send_wrapper(CONTINUE, i + 1)
            #send(x, i + 1)
            send_wrapper(x, i + 1)

            #reqs[i] = irecv(grads[i], i + 1)
            reqs[i] = irecv_wrapper(grads[i], i+1)
            set_a_req(reqs,i) # used only for gloo backend
        i = (i + 1) % (world_size - 1)

    print(f"master loop exiting,")
    print(f"gradient order: {grad_order}")

def master_cleanup(reqs,FINISH):
    # clean-up (tell other processes to shut down)
    i = 0
    finishes_sent = 0
    not_sent = [1 for i in range(world_size-1)]
    while True:
        if check_a_req(reqs[i]) and not_sent[i]:
            not_sent[i] = 0
            #send(FINISH, i + 1)
            send_wrapper(FINISH, i + 1)
            finishes_sent += 1
            if finishes_sent >= world_size - 1:
                break
        i = (i + 1) % (world_size - 1)

def master():

    x = torch.tensor([[1., -1.], [1., -1.]]).cuda()
    CONTINUE = torch.tensor([1]).cuda()
    FINISH = torch.tensor([0]).cuda()

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
        grad = 2*x + sigma*torch.randn(2,2).cuda()
        time.sleep(0.1*np.random.uniform())
        #time.sleep(0.1*global_rank)
        #req = isend(grad, 0)
        req = isend_wrapper(grad, global_rank)
        print_some(f"worker {global_rank} waiting...")
        req.wait()
        ngrads += 1
        #recv(continue_flag, 0)
        recv_wrapper(continue_flag, global_rank)
        if continue_flag[0] == 0:
            break
        #recv(x, 0)
        recv_wrapper(x, global_rank)
    print(f"worker {global_rank} exiting after computing {ngrads} gradients")

def worker():


    x = torch.zeros(2,2).cuda()
    continue_flag = torch.tensor([-1]).cuda()
    # initial send to everyone
    dist.broadcast(x, 0)
    worker_loop(x,continue_flag)

    #print(f"worker {global_rank} exiting with x = {x}")




################################################################################
# Main
################################################################################
if global_rank==0:
    master()
else:
    worker()


