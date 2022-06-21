################################################################################
# Master-worker synchronous set-up
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
def master_loop(d,max_iter,compute_grp):
    # loop to check for an irecv completed
    
    CONTINUE = torch.tensor([1])
    checkObjRate = 5
    
    z = torch.zeros(d)
    
    x = []
    w = []
    for _ in range(numSlices+1):
        x.append(torch.zeros(d))
        w.append(torch.zeros(d))

    gradw = []
    for _ in range(numSlices):
        gradw.append(torch.zeros(d))
    

    gradzphi = torch.zeros(d)
    sumphi = torch.zeros(1)

    for itr in range(max_iter):
    
        ###############
        # perform gather to get avx 
        ###############
        
        dist.gather(torch.zeros(d),x,group=compute_grp)
        avx = torch.zeros(d)
        for i in range(1,len(x)):
            avx += x[i]
        avx = avx/numSlices 

        ###############
        # get sumzgrads
        ###############
        
        gradzphi = torch.zeros(d)
        dist.reduce(gradzphi,0,group=compute_grp)

        gradznorm = torch.linalg.norm(gradzphi)**2

        ###############
        # compute gradwPhi 
        ###############
        gradwnorm = 0.
        for i in range(1,len(x)):
            gradw[i-1] = x[i] - avx 
            gradwnorm += torch.linalg.norm(gradw[i-1])**2 

        
        ###############
        # compute phi
        ###############
        phi = torch.zeros(1)
        dist.reduce(phi,0,group=compute_grp)

        ###############
        # compute alpha 
        ###############
        alpha = np.max([0,phi[0]])/(gradwnorm+gradznorm)
        

        ###############
        # Update z and ws
        ###############
        z = z - alpha*gradzphi
        
        for j in range(1,numSlices+1):
            w[j] = w[j] - alpha*gradw[j-1]

        dist.broadcast(z,0,group=compute_grp) 
        dist.scatter(w[0],w,group=compute_grp)

        if (itr+1)%checkObjRate == 0:
            print_some('iteration '+str(itr+1)+' sumphi = '+str(phi))
            dist.send(CONTINUE, world_size-1)
            dist.send(z,world_size-1)

        
    print(f"master loop exiting,")
    
 
################################################################################
# Worker code
################################################################################
def worker_loop(Ai,bi,max_iter,addDelay,scaleTime,procNum,compute_grp):
    d = Ai.shape[1]
    z = torch.zeros(d)
    wi = torch.zeros(d)
    nupdates = 0
    
    L = torch.linalg.norm(Ai,2)**2
    rho = L**(-1)
    #rho = 1e-7

    D=1

    for itr in range(max_iter):
        # compute 
        backtracking=True
        rho = 1.2*rho 
        while backtracking:
            xi = z - rho*(Ai.T@(Ai@z-bi) - wi)
            gradxi = Ai.T@(Ai@xi-bi)
            if (z-xi).dot(gradxi-wi) >= D*torch.linalg.norm(z-xi)**2:
                backtracking=False 
            else:
                rho = .8*rho 

        phi_i = (z-xi).dot(gradxi-wi)

        if addDelay:
            delayTime = scaleTime*np.random.uniform()*procNum**2
            time.sleep(delayTime)
        
        # send
        dist.gather(xi,dst=0,group=compute_grp)
        dist.reduce(gradxi,dst=0,group=compute_grp)
        dist.reduce(phi_i,dst=0,group=compute_grp)
        
        # receive 
        dist.broadcast(z,0,group=compute_grp) 
        dist.scatter(wi,group=compute_grp)

    #print(f"worker {global_rank} exiting")

################################################################################
# Monitor code
################################################################################
def monitor(A,b,algo,addDelay):
    '''
    monitor simply computes objective values and will run as the last process id
    '''
    actualStart = time.time()
    if addDelay:
        algo +='delay_'

    if os.path.exists(algo+'out.txt'):
        os.remove(algo+'out.txt')


    
    z = torch.zeros(A.shape[1])
    continue_flag = torch.tensor([-1])
    objvals = []
    times = []
    
    
    init_obj = 0.5*torch.linalg.norm(A@z-b)**2
    print('init objective = '+str(np.round(init_obj.item(),4)))
    targetTol = .5
    notReachedYet = True 
    FirstTime = True 
    while True:
        dist.recv(continue_flag, 0)

        if continue_flag[0] == 0:
            break

        dist.recv(z,0)
        if FirstTime:
            tstart = time.time()
            FirstTime = False 

        objval = 0.5*torch.linalg.norm(A@z-b)**2
        times.append(time.time()-tstart)
        print_monitor('objective = '+str(np.round(objval.item(),4))+' at time: '+str(np.round(times[-1],4)),algo)
        objvals.append(objval)
        if notReachedYet and (objval<targetTol*init_obj):
            tSuccess = time.time()-tstart
            print('target obj reduction reached at '+str(np.round(tSuccess,4)))
            notReachedYet = False 

        # compute objective value here
        #print_some('I monitor got z')
    print('monitor quitting, total run time: ',np.round(time.time()-actualStart,4))


def parseCMDLine():
    parser = argparse.ArgumentParser(description='distributed PS')
    parser.add_argument('--addDelay', action='store_true', default=False,
                        help='whether to add time delay to worker updates to simulate heteregeneity')
    parser.add_argument('--dist-backend', default='mpi', type=str,
                        help='distributed backend')
    parser.add_argument('--scaleTime', default=1., type=float,
                        help='scaling of time for the delay durations')
    parser.add_argument('--maxIter', default=10, type=int,
                        help='max iterations')
    parser.add_argument('--checkObjRate', default=10, type=int,
                        help='check objective val this often')

    args = parser.parse_args()
    return args 
          

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
        master_loop(d,max_iter,compute_grp)
        FINISH = torch.tensor([0])
        dist.send(FINISH, world_size-1) # close monitor 
        print("master exiting")
    elif global_rank < world_size-1:
        worker_loop(Ai,bi,max_iter,addDelay,scaleTime,global_rank,compute_grp)
    else:
        monitor(A,b,'sync_',addDelay)


