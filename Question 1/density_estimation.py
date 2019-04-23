#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 13:20:15 2019

@author: chin-weihuang
"""


from __future__ import print_function
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
import torch.nn.functional as F
from samplers import distribution4
from GANQ1 import *

# plot p0 and p1
plt.figure()
# empirical
xx = torch.randn(10000)
f = lambda x: torch.tanh(x*2+1) + x*0.75
d = lambda x: (1-torch.tanh(x*2+1)**2)*2+0.75
plt.hist(f(xx), 100, alpha=0.5, density=1)

plt.hist(xx, 100, alpha=0.5, density=1)
plt.xlim(-5,5)

# exact
xx = np.linspace(-5,5,1000)
N = lambda x: np.exp(-x**2/2.)/((2*np.pi)**0.5)
plt.plot(f(torch.from_numpy(xx)).numpy(), d(torch.from_numpy(xx)).numpy()**(-1)*N(xx))
plt.plot(xx, N(xx))


############### train a discriminator on distribution4 and standard gaussian
############### estimate the density of distribution4

#######--- INSERT YOUR CODE BELOW ---#######
 
batch_size = 512

# training of discriminator
D = Discriminator(input_size=1, hidden_size=64)
D = train_q4(model=D, loss_fn=JSD_loss_function_q4, num_epochs=20000, batch_size=batch_size,
             f0_samples=N(xx), f1_samples=iter(distribution4(1)))

with torch.no_grad():
    discriminator_xx = D(torch.tensor(xx, dtype=torch.float).view(1000, 1))

r = discriminator_xx.cpu().numpy() # evaluate xx using your discriminator; replace xx with the output
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.plot(xx,r)
plt.title(r'$D(x)$')

r = r.flatten()
estimate = np.array(N(xx))*np.divide(r, 1-r)

plt.subplot(1,2,2)
plt.plot(xx, estimate)
plt.plot(f(torch.from_numpy(xx)).numpy(), d(torch.from_numpy(xx)).numpy()**(-1)*N(xx))
plt.legend(['Estimated','True'])
plt.title('Estimated vs True')
plt.show()










