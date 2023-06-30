import torch.distributions as dist
import numpy as np

import math
import os
import h5py
from mpire import WorkerPool as Pool

from geobed.fwd_collection.avo import *


def data_likelihood_avo(samples, **kwargs):    
    std_data = 0.05
    data_likelihood = dist.Independent(dist.Normal(samples, torch.tensor(std_data)), 1)
    
    return data_likelihood


def generate_lookup_table(filename, prior_samples, offsets, forward_function, n_parallel = 5, transpose_order=(1,0,2)):

    if os.path.isfile(filename):
        print('loading lookup table from file')
        with h5py.File(filename, 'r') as f:
            # data = f['data'][:]
            prior_samples = f['prior'][:]
        # data = torch.from_numpy(data)
        prior_samples = torch.from_numpy(prior_samples)
    else:
        print('generating lookup table')
        def parallel_func(des):
            # be careful: prior samples is set at first definition of function
            return forward_function(des[None], prior_samples).numpy()
            
        print('starting parallel processing')
        with Pool(n_parallel) as pool:
            data = pool.map(parallel_func, [[des,] for des in offsets], progress_bar=True, concatenate_numpy_output=False)
        
        
        data = np.array(data).transpose(transpose_order)

        with h5py.File(filename, 'w') as f:
            f.create_dataset('data',  data=data)
            f.create_dataset('prior', data=prior_samples.numpy())
            
        print('dimensions: (n_prior, n_design, data_dim)')
        print(data.shape, data.dtype,  '\n')

        # quick check if all calculations succeded and the data is ok
        print(np.count_nonzero(data))
        print(np.count_nonzero(np.isnan(data)))
        
    return prior_samples

def nmc_partition(T):
    M = math.pow(T, 1/3)
    N = M*M
    return int(N), int(M)

def nmc_reuse_partiton(T):
    M = 0.5 * T
    N = 0.5 * T
    return int(N), int(M)

def variational_partiton(T):
    N = 0.2 * T
    M = 0.8 * T
    return int(N), int(M)
