################################################################################
# Parallel GD
################################################################################
import argparse
import torch
import sys
import os
#sys.path.append("../")
from distributed import init_workers
import torch.distributed as dist
import threading
import time
import numpy as np

# local
from syncCentralPS import monitor,parseCMDLine 

################################################################################
# Print function useful for debugging async parallel programs
################################################################################
def print_some(thing):
    original_stdout = sys.stdout  # Save a reference to the original standard output
    with open('print_output.txt', 'a') as f:
        sys.stdout = f  # Change the standard output to the file we created.
        print(thing)
        sys.stdout = original_stdout  # Reset the standard output to its original value

def print_monitor(thing,algo):
    original_stdout = sys.stdout  # Save a reference to the original standard output
    with open(algo+'out.txt', 'a') as f:
        sys.stdout = f  # Change the standard output to the file we created.
        print(thing)
        sys.stdout = original_stdout  # Reset the standard output to its original value

################################################################################
# Master code
################################################################################
def master_loop(d,max_iter,compute_grp,rho):
    # loop to check for an irecv completed
    
    CONTINUE = torch.tensor([1])
    FINISH = torch.tensor([0])

    checkObjRate = 5
    
    z = torch.zeros(d)
    x = torch.zeros(d)

    

    t = 1. 

    for itr in range(max_iter):
    
        ###############
        # perform reduce to get gradz 
        ###############
        gradz = torch.zeros(d)
        dist.reduce(gradz,0,group=compute_grp)
        
        
        xplus = z - rho*gradz

        tplus = 0.5*(1+np.sqrt(1+4*t**2))
        alpha = (t-1)/tplus 
        #alpha = 0.
        t = tplus 

        z = xplus + alpha*(xplus-x)

        x = torch.clone(xplus)
        
        dist.broadcast(z,0,group=compute_grp)


        
        if (itr+1)%checkObjRate == 0:
            print_some('iteration GD '+str(itr+1))
            dist.send(CONTINUE, world_size-1)
            dist.send(z,world_size-1)

        
    print(f"master loop exiting,")
    
 
################################################################################
# Worker code
################################################################################
def worker_loop(Ai,bi,max_iter,addDelay,scaleTime,procNum,compute_grp):
    d = Ai.shape[1]
    z = torch.zeros(d)
    nupdates = 0
    flag = torch.tensor([-1])
    
    

    for itr in range(max_iter):
        # compute     
        gradz = Ai.T@(Ai@z-bi)

        if addDelay:
            delayTime = scaleTime*np.random.uniform()*procNum**2
            time.sleep(delayTime)

        # send
        dist.reduce(gradz,dst=0,group=compute_grp)
        
        
        # receive
        dist.broadcast(z,0,group=compute_grp)
        
        


        

    



if __name__ == "__main__":
    ################################################################################
    # command line arguments
    ################################################################################
    args = parseCMDLine()
    addDelay = args.addDelay
    distributed_backend = args.dist_backend 
    scaleTime = args.scaleTime
    max_iter = args.maxIter 

    ################################################################################
    # Distributed set-up
    ################################################################################
    global_rank, world_size = init_workers(distributed_backend)

    if (global_rank==0) and os.path.exists('print_output.txt'):
        os.remove('print_output.txt')

    print(f"my global rank is {global_rank}")
    if global_rank==0:
        print(f"world size: {world_size}")

    ################################################################################
    # Create separate group excluding monitor 
    ################################################################################
    compute_grp = dist.new_group(list(range(world_size-1)))

    ################################################################################
    # Lasso set-up
    ################################################################################
    
    #m = 18*200
    #d = 10000
    m = 18*50
    d = 2000

    numSlices = world_size - 2 # 1 master at proc 0 and monitor at proc world_size-1
    sliceSz = int(m/numSlices) # better just make sure this is an integer
    print_some('sliceSz = '+str(sliceSz))
    print_some('numSlices = '+str(numSlices))
    if global_rank==0:
        A = torch.load('A.torch')
        b = torch.load('b.torch')

        L = torch.linalg.norm(A,2)**2
        print_some('Global Lipschitz constant L: '+str(L))
        print_some(L**(-1))
        rho = L**(-1)
        for proc in range(1,world_size-1):
            Aslice = A[(proc-1)*sliceSz:proc*sliceSz]
            bslice = b[(proc-1)*sliceSz:proc*sliceSz]

            dist.send(Aslice,proc)
            dist.send(bslice,proc)
        

    elif global_rank < world_size-1:

        Ai = torch.empty((sliceSz,d))
        bi = torch.empty(sliceSz)
        
        #print_some('process '+str(global_rank)+'starting recvs')
        dist.recv(Ai,0)
        #print_some('process '+str(global_rank)+'got Ai')
        dist.recv(bi,0)
        #print_some('process '+str(global_rank)+'got bi')
        
        #print('proc 1 received Ai with shape: ',Ai.shape, 'and bi with shape:',bi.shape)

    else:
        A = torch.load('A.torch')
        b = torch.load('b.torch')
        

    ################################################################################
    # Main
    ################################################################################
    

    if global_rank==0:
        master_loop(d,max_iter,compute_grp,rho)
        FINISH = torch.tensor([0])
        dist.send(FINISH, world_size-1) # close monitor 
        print("master exiting")
    elif global_rank < world_size-1:
        worker_loop(Ai,bi,max_iter,addDelay,scaleTime,global_rank,compute_grp)
    else:
        monitor(A,b,'parallelGD_',addDelay)


