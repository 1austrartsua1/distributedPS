import torch

A = torch.load('A.torch')
b = torch.load('b.torch')
(m,d) = A.shape 
z = torch.zeros(d)
wi = torch.zeros(d)
#L = torch.linalg.norm(A,2)**2
#rho = L**(-1)
rho = 1e-7
MAX_UPDATES = 10000
CHECK_EVERY = 1000
D=1
for i in range(1,MAX_UPDATES+1):
    backtracking = True 
    rho = 1.2*rho 
    while backtracking:
        xi =  z - rho*(A.T@(A@z - b) - wi)
        gradxi = A.T@(A@xi - b)
        if (z-xi).dot(gradxi) >= D*torch.linalg.norm(z-xi)**2:
            backtracking=False 
        else:
            rho = .8*rho 

    phi = (z-xi).dot(gradxi-wi)
    alpha = phi/(torch.linalg.norm(gradxi)**2)
    #alpha=1e-7
    z = z - alpha*gradxi 
    if i%CHECK_EVERY==0:
        obj = 0.5*torch.linalg.norm(A@z-b)**2
        print(obj)
        #print(phi)


