import os
import math
import numpy as np
import torch
import torch.distributions as dist

import h5py
from mpire import WorkerPool as Pool

import geobed.fwd_collection.raytracer as raytracer


def construct_covmat(theta, ratio, scaling): 
	#TODO: Rewrite in torch
    theta = -np.radians(theta)
    ratio = ratio
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    S = scaling * np.diag([ratio, 1])
    L = S**2
    return torch.tensor(R@L@R.T, dtype=torch.float32)

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

def load_marmousi_file(filename):
    with np.load(filename, allow_pickle=True) as marmousi_centersection:    
        x = marmousi_centersection['x']
        z = marmousi_centersection['z']
        marmousi_vp = marmousi_centersection['data']
    return x, z, marmousi_vp

def gmm_prior_dist():
    mu1=torch.tensor([9.37, 1.91])*1e3
    mu2=torch.tensor([10.1, 2.0])*1e3
    mu3=torch.tensor([10.85, 2.01])*1e3
    mu_list = [mu1, mu2, mu3]

    cov1 = construct_covmat(10,  4, 0.12*1e3)
    cov2 = construct_covmat(15,  6, 0.1*1e3)
    cov3 = construct_covmat(5, 10, 0.08*1e3)
    cov_list = [cov1, cov2, cov3]

    mix = dist.Categorical(torch.tensor([0.3, 0.3, 0.3]))
    comp = dist.MultivariateNormal(torch.stack(mu_list,axis=0),
                                covariance_matrix=torch.stack(cov_list, axis=0))

    prior_dist = dist.MixtureSameFamily(mix, comp)
    return prior_dist

def data_likelihood_gmm(data, **kwargs):
    # Abakumov et al 2020
    std_model = 0.02
    return dist.Independent(dist.Normal(data, std_model), 1)

def calculate_traveltime_table(x, z, vp_model, prior_samples, designs, filename, n_parallel=1):
    
    dx = x[1] - x[0]
    dz = z[1] - z[0]
    
    tt_H = raytracer.TTHelper()
    tt_H.set_model( 
        x= x,  y= np.array([0]),  z=z,
        dx=dx, dy=          1.0, dz=dz,
        velocity_model=vp_model)
    
    if os.path.isfile(filename):
        with h5py.File(filename, 'r') as f:
            # tt_array = f['data'][:]
            prior_samples = f['prior'][:]
        
        # tt_array = torch.from_numpy(tt_array)
        prior_samples = torch.from_numpy(prior_samples)

    else:    
        def parallel_func(des):
            # be careful: prior samples is set at first definition of function
            return tt_H.calculate_tt_diff(des, prior_samples)
        
        tt_array = []
        print('starting parallel processing')
        with Pool(n_parallel) as pool:
            out = pool.map(parallel_func, [[des,] for des in designs], progress_bar=True, concatenate_numpy_output=False)
            tt_array = np.array(out).T
            
        print(tt_array.shape)
        
        mask = ~np.isnan(tt_array).any(axis=1)
        tt_array      = tt_array[mask]
        prior_samples = prior_samples[mask]
        
        tt_array = tt_array[:, :, None]
        
        with h5py.File(filename, 'w') as f:
            f.create_dataset("data",  data=tt_array)
            f.create_dataset("prior", data=prior_samples)     
        
        print('dimensions: (n_prior, n_design, data_dim)')
        print(tt_array.shape, tt_array.dtype,  '\n')

        # # quick check if all calculations succeded and the data is ok
        # print(np.count_nonzero(tt_array))
        # print(np.count_nonzero(np.isnan(tt_array)))
        
        if np.count_nonzero(np.isnan(tt_array)) > 0:
            raise ValueError('NaN in tt_array')

        del tt_array

    return prior_samples


def check_batch_epoch(n_batch, n_epochs, N, M):
    if type(n_batch) != int:
        if type(n_batch) == float:
            n_batch = int(n_batch)
        elif callable(n_batch):
            n_batch = n_batch(**{'N':N, 'M':M})
        else:
            raise ValueError('n_batch must be int, float or callable')
    
    print('n_batch: ', n_batch)
    
    if type(n_epochs) != int:
        if type(n_epochs) == float:
            n_epochs = int(n_epochs)
        elif callable(n_epochs):
            n_epochs = n_epochs(**{'N':N, 'M':M, 'n_batch':n_batch})
        else:
            raise ValueError('n_epochs must be int, float or callable')
    
    print('n_epochs: ', n_epochs)
    
    return n_batch, n_epochs


def slowness_grid(x, z, method='gradient', edges=True, **kwargs):
    if method == 'gradient':
        
        if edges:
            N_x = x.shape[0] - 1
            N_z = z.shape[0] - 1
        else:
            N_x = x.shape[0]
            N_z = z.shape[0]
            
        coarsness = kwargs.get('coarsness', 1)
        
        if N_z % coarsness != 0:
            raise ValueError('N_z must be a multiple of N_steps')
        else:
            N_c = N_z // coarsness
        
        vel = np.empty((N_c,))

        v_1  = kwargs['v_top']
        v_2  = kwargs['v_bottom'] 
        v_step = (v_1 - v_2) / (N_c - 1)

        for n in range(N_c):
            vel[n] = v_1 - n * v_step
            
        if coarsness > 1:
            vel = np.repeat(vel, coarsness)
            
        vel = np.tile(vel, N_x).reshape(N_x, N_z)

        return 1/vel

    if method == 'array':
        if 'velocity' in kwargs:
            return 1/kwargs['velocity']
        if 'slowness' in kwargs:
            return kwargs['slowness']
        else:
            raise ValueError('No velocity array provided')    
    else:
        raise ValueError('Unknown method')

