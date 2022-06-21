import torch 
'''
m = 18*250
d = 10000
'''
m = 18*50
d = 2000

A = torch.rand((m,d))
b = torch.rand(m)
torch.save(A,'A.torch')
torch.save(b,'b.torch')
