import numpy as np 
import torch 

A = torch.load('A.torch')
b = torch.load('b.torch')
(m,d) = A.shape 
z = torch.zeros(d)
x = torch.zeros(d)

maxIter = 1000


L = torch.linalg.norm(A,2)**2
rho = L**(-1)
t = 1
eps = 1e-3
bt = 0
for itr in range(maxIter):
    gradz = A.T@(A@z - b)
    #rho = 1.1*rho 
    f0 = 0.5*np.linalg.norm(A@z - b)**2
    while True:
        xplus = z - rho*gradz 
        
        f = 0.5*np.linalg.norm(A@xplus - b)**2
        break 
        bt+=1
        #print(bt)
        if f<f0:
            break 
        else:
            rho = .8*rho 
    
    tplus = 0.5*(1+np.sqrt(1+4*t**2))
    alpha = (t-1)/tplus 
    t = tplus 
    #alpha = 0.
    z = xplus + alpha*(xplus-x)
    x = np.copy(xplus)


    print(f) 