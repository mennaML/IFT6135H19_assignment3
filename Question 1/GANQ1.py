import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import samplers as sampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def KL(p, q):
    if np.sum(p)!=1 or np.sum(q)!=1 :
            raise Exception("input in not a valid pdf")

    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)


    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def JS(p, q):
    return 0.5*(KL(p, (p+q)/2) + KL(q, (p+q)/2))

def JSD_loss_function_q4(Dx, Dy):
    return -1*(torch.mean(torch.log(Dx)) + torch.mean(torch.log(1 - Dy)))

def JSD_loss_function(Dx, Dy):
    return -1*(torch.mean(torch.log(Dx)) + torch.mean(torch.log(1 - Dy)))

def WD_loss_function(Tx, Ty, gradient_penalty=True):
    return -1*(torch.mean(Tx) - torch.mean(Ty))

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
            nn.Linear(hidden_size, hidden_size),
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
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    p_gen = sampler.distribution1(0, batch_size=batch_size)
    q_generator = sampler.distribution1(phi, batch_size=batch_size)
    a_generator = sampler.distribution2(batch_size=batch_size)

    losses = []
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        x = torch.tensor(next(p_gen), dtype=torch.float)
        y = torch.tensor(next(q_generator), dtype=torch.float)
        
        x.to(device)
        y.to(device)

        Dx = model(Variable(x))
        Dy = model(Variable(y))
        loss = loss_fn(Dx, Dy)

        if loss_fn == WD_loss_function:
            #sample a and compute z
            a = torch.tensor(next(a_generator), dtype=torch.float) 
            z = a*x + (1-a)*y
   
            z.requires_grad_(True)
            Dz = model(z)
            Dz.requires_grad_(True)
            loss.backward(retain_graph=True, create_graph=True)

            grad_Tz = torch.autograd.grad(Dz.sum(), z, create_graph=True)[0]
            norm_gradient = torch.norm(grad_Tz, p=2, dim=-1)

            penalty = (norm_gradient - 1).pow(2).mean()
            loss += lmdba*penalty

            #loss.backward(retain_graph=True, create_graph=True)
            
        
        loss.backward()

        losses.append(loss)
        optimizer.step()
        
        if( epoch % int(num_epochs/10)) == (int(num_epochs/10)-1) :
            print( "Epoch %6d. Loss %5.3f" % ( epoch+1, loss ) )
        
    print( "Training complete for discriminator with parameter phi=", phi)
    return model

def eval(model, batch_size, phi):
    q_generator = sampler.distribution1(phi, batch_size=batch_size)

    model.eval()
    q = torch.tensor(next(q_generator), dtype=torch.float)

    return model(q)


def train_q4(model, loss_fn, num_epochs, batch_size, f0_samples, f1_samples):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    p_gen = sampler.distribution4(batch_size=batch_size)
    q_gen = sampler.distribution3(batch_size=batch_size)

    losses = []
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        x = torch.tensor(next(p_gen), dtype=torch.float)
        y = torch.tensor(next(q_gen), dtype=torch.float)

        x.to(device)
        y.to(device)

        Dx = model(Variable(x))
        Dy = model(Variable(y))
        loss = loss_fn(Dx, Dy)

        loss.backward()

        losses.append(loss)
        optimizer.step()

        if (epoch % int(num_epochs / 10)) == (int(num_epochs / 10) - 1):
            print("Epoch %6d. Loss %5.3f" % (epoch + 1, loss))

    print("Training complete for discriminator (Q1.4)",)
    return model


def question_1_3_JSD():
    phi = np.arange(-1, 1.1, step=0.1)
    eval_size = 100
    uniformsJSD = []
    Dy = torch.zeros((eval_size, 21))
    # training
    for i in range(21):
        D = Discriminator(input_size=2, hidden_size=64)
        D = train(model=D, loss_fn=JSD_loss_function, num_epochs=10000, batch_size=512, phi=phi[i])
        Dy[:, i:i + 1] = eval(D, batch_size=eval_size, phi=phi[i])

    # computing the loss
    for i in range(21):
        uniformsJSD.append(np.log(2) - 0.5 * JSD_loss_function(Dy[:, 10], Dy[:, i]))

    plt.plot(phi, uniformsJSD)
    plt.xlabel('Phi')
    plt.ylabel('JSD')
    plt.show()


def question_1_3_WSD():
    phi = np.arange(-1, 1.1, step=0.1)
    uniformsWD = []
    eval_size = 100
    Dy = torch.zeros((eval_size, 21))

    # training
    for i in range(21):
        D = Discriminator(input_size=2, hidden_size=64)
        D.to(device)
        D = train(model=D, loss_fn=WD_loss_function, num_epochs=6000, batch_size=512, phi=phi[i])
        Dy[:, i:i + 1] = eval(D, batch_size=eval_size, phi=phi[i])

    # computing the loss
    for i in range(21):
        uniformsWD.append(-WD_loss_function(Dy[:, 10], Dy[:, i]))

    np.save('WSD_plt_data.npy', uniformsWD)

    plt.plot(phi, uniformsWD)
    plt.xlabel('Phi')
    plt.ylabel('WD')
    plt.show()

###############################################################################
#
# Q1.3: training discriminator on JSD and WD loss
#
###############################################################################

if __name__ == '__main__':

    question_1_3_JSD()
    question_1_3_WSD()


###############################################################################
#
# Q1.4: Implemented in density_estimation.py
#
###############################################################################