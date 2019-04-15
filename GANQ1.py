import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import samplers as sampler

def KL(p, q):
    if np.sum(p)!=1 or np.sum(q)!=1 :
            raise Exception("input in not a valid pdf")

    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)


    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def JS(p, q):
    return 0.5*(KL(p, (p+q)/2) + KL(p, (p+q)/2))

def JSD_loss_function(Dx, Dy, p, q):
    #E_pDx = torch.mean(torch.log(Dx))
    #E_qDy = torch.mean(torch.log(1 - Dy))
    
    return -1*(np.log(2) + 0.5*torch.mean(torch.log(Dx)) + 0.5*torch.mean(torch.log(1 - Dy)))

def W_loss_function(Tx, Ty, p, q):
    E_pTx = np.mean(np.where(p != 0, p * Tx, 0))
    E_qTy = np.mean(np.where(q != 0, q * Ty, 0))

    return -1*(E_pTx - E_qTy)
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Discriminator, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.weights_init()
    def weights_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight.data)
            torch.nn.init.xavier_uniform(m.bias.data)
    
    def forward(self, x):
        # flatten input if needed
        x = x.view(x.size(0)*x.size(1))
        x = self.layers(x)
        return torch.sigmoid(x)



def train(model, loss_fn, num_epochs, batch_size):
    phi = 1
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    p_generator = sampler.distribution1(0, batch_size=batch_size)
    q_generator = sampler.distribution1(phi, batch_size=batch_size)

    losses  = []
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        
        p = torch.tensor(next(p_generator), dtype=torch.float)
        q = torch.tensor(next(q_generator), dtype=torch.float)


        Dx = model(Variable(p))
        Dy = model(Variable(q))
        loss = JSD_loss_function(Dx, Dy, p, q)

        if loss_fn == W_loss_function:
            print('agg grad')

        loss.backward()

        losses.append(loss)
        optimizer.step()
        
        #if( epoch % int(num_epochs/10)) == (int(num_epochs/10)-1) :
        print( "Epoch %6d. Loss %5.3f" % ( epoch+1, loss ) )
        
    print( "Training complete" )
        



D = Discriminator(input_size=512*2, hidden_size=64)
#D.apply(weights_init)
train(model=D, loss_fn=JSD_loss_function, num_epochs=100, batch_size=512)