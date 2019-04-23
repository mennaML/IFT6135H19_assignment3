import torch
from dataset import Binary_MNIST_DS
from VAE_model import VAE
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from tqdm import tqdm
import random
import os

import numpy as np
import importance_sampling as sampler



num_epochs = 20
batch_size = 32
learning_rate = 3*(10**-4)
        

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}

train_dataset = Binary_MNIST_DS('data/binarized_mnist_train.amat', transform=transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)

valid_dataset = Binary_MNIST_DS('data/binarized_mnist_valid.amat', transform=transforms.ToTensor())
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, **kwargs)

test_dataset = Binary_MNIST_DS('data/binarized_mnist_test.amat', transform=transforms.ToTensor())
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)


def fix_seed(seed):
    '''
    Fix the seed.

    Parameters
    ----------
    seed: int
        The seed to use.

    '''
    print('pytorch/random seed: {}'.format(seed))
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Reconstruction + KL divergence 
def loss_function(recon_x, x, mu, logvar):
    
    #loss = - ELBO = BCE + KLD
    
    BCE = F.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), reduction='sum')

    # See https://arxiv.org/abs/1312.6114 - Appendix B
    # -KLD = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(model, optimizer, epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(tqdm(train_loader)):
        data = data.to(device)
        optimizer.zero_grad()
        reconstructed_batch, mu, logvar = model(data)
        loss = loss_function(reconstructed_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))
       

def evaluate_elbo(model, dataloader, epoch=0, verbose=False):

    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader)):
            data = data.to(device)
            reconstructed_batch, mu, logvar = model(data)
            l = loss_function(reconstructed_batch, data, mu, logvar).item()
            #print('loss_function', l)
            test_loss += l
    test_loss /= len(dataloader.dataset)
    elbo = -test_loss
    if verbose:
        print('====> Epoch: {} Average ELBO: {:.4f}'.format(epoch, elbo))
    
    return -test_loss

def estimate_log_density(model, data_loader, num_samples):
    
    model.eval()
    batches = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader)):
            batch = batch.to(device)
            batches.append(sampler.estimate_batch_log_density(model, batch, num_samples, device).cpu().numpy())
            all_log_p_x = np.concatenate(batches)
    return np.mean(all_log_p_x)

if __name__ == "__main__":
    
    fix_seed(1)
    
    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(1, num_epochs + 1):
        train(model, optimizer, epoch)
        evaluate_elbo(model, valid_loader, epoch, verbose=True)

    logpx_valid = estimate_log_density(model, valid_loader, 200)
    print("====> Validation set log p(x) approximation: %.2f" % logpx_valid)
    
    elbo_test = evaluate_elbo(model, test_loader)
    print('====> Test set ELBO: {:.4f}'.format(elbo_test))
    
    logpx_test = estimate_log_density(model, test_loader, 200)
    print("====> Test set log p(x) approximation: %.2f" % logpx_test)

