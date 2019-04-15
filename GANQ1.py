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

def JSD_loss_function(Dx, Dy, p=0, q=0):
    return -1*(0.5*torch.mean(torch.log(Dx)) + 0.5*torch.mean(torch.log(1 - Dy)))

def WD_loss_function(Tx, Ty, p, q):
    #E_pTx = np.mean(np.where(p != 0, p * Tx, 0))
    #E_qTy = np.mean(np.where(q != 0, q * Ty, 0))

    return -1*(np.mean(Tx) - np.mean(Ty))

###############################################################################
#
# Discriminator Class
#
###############################################################################

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Discriminator, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, hidden_size//2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size//2, 1),
            nn.Sigmoid(), 
        )
        self.weights_init()
    def weights_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight.data)
            torch.nn.init.xavier_uniform(m.bias.data)
    
    def forward(self, x):
        x = self.layers(x)
        return x



def train(model, loss_fn, num_epochs, batch_size, phi, lmdba=10):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005)
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
        loss = loss_fn(Dx, Dy, p, q)

        if loss_fn == WD_loss_function:
            #sample a and compute z
            Dz = model(Variable(p))
            loss.backward(retain_graph=True, create_graph=True)
            grad_Tz = torch.autograd.grad(Dz, z, retain_graph=True)
            loss += torch.norm(grad_Tz, p=2, dim=-1)
            print('agg grad')
        
        loss.backward()

        losses.append(loss)
        optimizer.step()
        
        if( epoch % int(num_epochs/10)) == (int(num_epochs/10)-1) :
            print( "Epoch %6d. Loss %5.3f" % ( epoch+1, loss ) )
        
    print( "Training complete" )
    return model
        


def eval(model, batch_size, phi):

    #p_generator = sampler.distribution1(0, batch_size=batch_size)
    q_generator = sampler.distribution1(phi, batch_size=batch_size)

    model.eval()
        
    #p = torch.tensor(next(p_generator), dtype=torch.float)
    q = torch.tensor(next(q_generator), dtype=torch.float)

    return model(Variable(q))


###############################################################################
#
# Q1.3: training 21 models 
#
###############################################################################
if __name__ == '__main__':
    phi = np.arange(-1,1.1,step=0.1)
    uniformsJSD = []
    uniformsWD = []
    Dy = torch.zeros((100,21))
    #Dy = []
    #Usining Jensen-shannon loss
    for i in range(21):
        D = Discriminator(input_size=2, hidden_size=64)
        D = train(model=D, loss_fn=JSD_loss_function, num_epochs=6000, batch_size=512, phi=phi[i])
        Dy[:, i:i+1] = eval(D, batch_size=100, phi=phi[i])

    for i in range(21):
        uniformsJSD.append(np.log(2) - JSD_loss_function(Dy[:,10], Dy[:,i]))

    plt.plot(phi, uniformsJSD)
    plt.show()
    input()
    plt.close()


    

    ###############################################################################
    #
    # Q1.4: 
    #
    ###############################################################################