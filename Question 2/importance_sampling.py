import numpy as np
import torch
import torch.distributions as tdist
from torch.nn import functional as F

# Generates z_ik, log_p_zik and log_q_zik
def generate_samples(model, x_batch, num_samples, device):

    mean_batch, logvar_batch = model.encode(x_batch)
    
    mv_normal = tdist.multivariate_normal.MultivariateNormal
    
    p_z_dist = mv_normal(torch.zeros(mean_batch.size(1)).to(device), 
                         torch.eye(mean_batch.size(1)).to(device))
    
    z_samples = torch.empty((mean_batch.size(0), 
                             num_samples, mean_batch.size(1)), device=device)
    log_q_z = torch.empty((mean_batch.size(0), num_samples), device=device)
    log_p_z = torch.empty((mean_batch.size(0), num_samples), device=device)
    
    for i in range(len(x_batch)):
        q_z_dist = mv_normal(mean_batch[i], 
                             torch.diag(torch.exp(0.5*logvar_batch[i])))
        z_i = q_z_dist.sample((num_samples,))
        log_q_z_i = q_z_dist.log_prob(z_i)
        log_p_z_i = p_z_dist.log_prob(z_i)
        
        z_samples[i] = z_i
        log_q_z[i] = log_q_z_i
        log_p_z[i] = log_p_z_i
        

    return z_samples, log_p_z, log_q_z

# Calculates the importance sampling approximation of log_p_x over a batch
def estimate_batch_log_density(model, x_batch, num_samples, device):

    z_samples, log_p_z, log_q_z = generate_samples(model, x_batch, num_samples, device)

    result = torch.empty((len(x_batch), ), device=device)
    for i in range(len(x_batch)):

        x_predict = model.decode(z_samples[i])
        log_p_z_i = log_p_z[i]
        log_q_z_i = log_q_z[i]
        
        log_p_x_z_i = torch.empty((num_samples, ), device=device)
        
        for k in range(num_samples):
            log_p_x_z_ik = -F.binary_cross_entropy(x_predict[k].view(-1, 784), 
                                                   x_batch[i].view(-1, 784), 
                                                   reduction='sum') 
            log_p_x_z_i[k] = log_p_x_z_ik.item()
            
        logsum = log_p_x_z_i + log_p_z_i - log_q_z_i
        
        logpx = -np.log(num_samples) + torch.logsumexp(logsum, 0)
        result[i] = logpx

    return result
