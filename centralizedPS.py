################################################################################
# Master-worker async polling set-up
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
# command line arguments
################################################################################

args = parseCMDLine()
addDelay = args.addDelay
distributed_backend = args.dist_backend 
scaleTime = args.scaleTime
max_iter = args.maxIter 
checkObjRate = args.checkObjRate 

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
def master_setup(d):
    # initial setup of recv from everyone
    reqs = []
    xandGradxs = []
    for i in range(1, world_size-1):
        xandGradxs.append(torch.zeros(2*d))
        reqs.append(dist.irecv(xandGradxs[-1], i))
        set_a_req(reqs, i - 1)  # used only for gloo backend
        
    return reqs,xandGradxs

def master_loop(reqs,xandGradxs,CONTINUE,z,w,MAX_UPDATES,checkObjRate):
    # loop to check for an irecv completed
    
    updates = 0
    
    d = z.shape[0]
    xold = []
    gradzOld = []
    gradw = []
    phiold = []
    xs = []
    gradxs = []
    computed_once = []
    for _ in range(numSlices):
        xold.append(torch.zeros(d))
        gradzOld.append(torch.zeros(d))
        gradw.append(torch.zeros(d))
        xs.append(torch.zeros(d))
        gradxs.append(torch.zeros(d))
        phiold.append(torch.zeros(1))
        computed_once.append(False)
        

    avx = torch.zeros(d)
    sumzgrads = torch.zeros(d)
    sumphi = torch.zeros(1)

    i = 0
    update_order = []
    while True:
        if check_a_req(reqs[i]):
            computed_once[i] = True 
            ###############
            # extract x and gradx from package recieved from worker 
            ###############
            
            xi = xandGradxs[i][:d]
            
            gradxi = xandGradxs[i][d:]

            xs[i] = torch.clone(xi)
            gradxs[i] = torch.clone(gradxi)
            
            ###############
            # compute average 
            ###############
            avx = avx - (1/numSlices)*xold[i] + (1/numSlices)*xi
            xold[i] = torch.clone(xi)

            ###############
            # compute gradzPhi 
            ###############
            

            sumzgrads = sumzgrads - gradzOld[i] + gradxi 
            gradzOld[i] = torch.clone(gradxi)
            

            gradznorm = torch.linalg.norm(sumzgrads)**2
            ###############
            # compute gradwPhi 
            ###############
            gradw[i] = xi - avx 

            gradwnorm = 0.
            for j in range(numSlices):
                gradwnorm += torch.linalg.norm(gradw[j])**2 

            ###############
            # compute phi
            ###############
            
            
            '''
            phi_i = (z-xi).dot(gradxi - w[i])
            sumphi = sumphi - phiold[i] + phi_i        
            phiold[i] = torch.clone(phi_i)
            '''
            
            sumphi = 0.
            for j in range(numSlices):
                sumphi += (z-xs[j]).dot(gradxs[j] - w[j])

            
            ###############
            # compute alpha 
            ###############
            alpha = np.max([0,sumphi])/(gradwnorm+gradznorm)


            properInitial = False 
            if properInitial and (sum(computed_once)<numSlices):
                alpha = 0.
            

            ###############
            # Update z and ws
            ###############
            
            
            z = z - alpha*sumzgrads
            

            for j in range(numSlices):
                w[j] = w[j] - alpha*gradw[j]

            

            updates += 1

            if updates%checkObjRate == 0:
                print_some('iteration '+str(updates)+' sumphi = '+str(sumphi))
                #print_some('alpha '+str(alpha))
                dist.send(CONTINUE, world_size-1)
                dist.send(z,world_size-1)

            update_order.append(i+1)
            if updates >= MAX_UPDATES:
                # send message to others to exit...
                break
            #send(CONTINUE, i + 1)
            
            dist.send(CONTINUE, i + 1)
            #send(x, i + 1)
            
            dist.send(z, i + 1)
            
            dist.send(w[i], i + 1)

            
            
            reqs[i] = dist.irecv(xandGradxs[i], i+1)
            set_a_req(reqs,i) # used only for gloo backend
            
            
        i = (i + 1) % (world_size - 2)

    print(f"master loop exiting,")
    print(f"update order: {update_order}")

def master_cleanup(reqs,FINISH):
    # clean-up (tell other processes to shut down)
    i = 0
    finishes_sent = 0
    not_sent = [1 for i in range(world_size-2)]
    while True:
        if check_a_req(reqs[i]) and not_sent[i]:
            not_sent[i] = 0
            #send(FINISH, i + 1)
            dist.send(FINISH, i + 1)
            finishes_sent += 1
            if finishes_sent >= world_size - 2:
                break
        i = (i + 1) % (world_size - 2)

    dist.send(FINISH, world_size-1) # close monitor 


def master(d,numSlices,world_size,MAX_UPDATES,checkObjRate):
    
    z = torch.zeros(d)
    w = torch.zeros((numSlices,d))

    CONTINUE = torch.tensor([1])
    FINISH = torch.tensor([0])

    reqs,xandGradxs = master_setup(d)
    
    master_loop(reqs,xandGradxs,CONTINUE,z,w,MAX_UPDATES,checkObjRate)
    
    master_cleanup(reqs,FINISH)

    print(f"master exiting")
    
################################################################################
# Worker code
################################################################################
def worker_loop(z,wi,Ai,bi,continue_flag,addDelay,procNum,scaleTime):
    nupdates = 0
    
    L = torch.linalg.norm(Ai,2)**2
    rho = .9*L**(-1)
    #rho = 1e-7

    D=1

    while True:
        backtracking=True
        rho = 1.2*rho 
        while backtracking:
            xi = z - rho*(Ai.T@(Ai@z-bi) - wi)
            gradxi = Ai.T@(Ai@xi-bi)
            #break 
            if (z-xi).dot(gradxi-wi) >= D*torch.linalg.norm(z-xi)**2:
                backtracking=False 
            else:
                rho = .8*rho 

        xiandGradxi = torch.cat([xi,gradxi])
        if addDelay:
            delayTime = scaleTime*np.random.uniform()*procNum**2
            time.sleep(delayTime)
        
        req = dist.isend(xiandGradxi, 0)
        #print_some(f"worker {global_rank} waiting...")
        req.wait()
        nupdates += 1
        dist.recv(continue_flag, 0)
        if continue_flag[0] == 0:
            break
        #recv(x, 0)
        dist.recv(z, 0)
        dist.recv(wi, 0)
    print(f"worker {global_rank} exiting after {nupdates} updates")

def worker(Ai,bi,addDelay,procNum,scaleTime):

    z = torch.zeros(d)
    wi = torch.zeros(d)

    continue_flag = torch.tensor([-1])

    worker_loop(z,wi,Ai,bi,continue_flag,addDelay,procNum,scaleTime)

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
# Lasso set-up
################################################################################


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
    master(d,numSlices,world_size,max_iter,checkObjRate)
elif global_rank < world_size-1:
    worker(Ai,bi,addDelay,global_rank,scaleTime)
else:
    monitor(A,b,'async_',addDelay)


