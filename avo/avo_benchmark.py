import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import random

import numpy as np
import matplotlib.pylab as plt
plt.rc('font', family='sans-serif')

from tqdm import tqdm

import torch
import torch.distributions as dist

import pandas as pd
import h5py

from geobed import BED_discrete
import geobed.fwd_collection.raytracer as raytracer

from utils import *
from geobed.guides import GMM_guide, MDN_guide, FullyConnected

# model prior
a_1_model = torch.tensor(2750.0)
d_model   = torch.tensor(500.0 )

a_2_mean = torch.tensor([3750.0])
a_2_std = torch.tensor([300.0])
prior_dist = dist.Independent(dist.Normal(a_2_mean, a_2_std), 1)

def forward_function(offset, a_2):
    return calculate_avo(offset, a_1_model, a_2, d_model)

data_likelihood = data_likelihood_avo

n_design_points = 200
design_names = [str(i) for i in range(n_design_points)]
offsets      = torch.linspace(100.0, 3500.0, n_design_points)

n_benchmarks = 10
n_random_benchmarks = 1000
n_parallel = 34

design_budget = 4

n_samples_min_exponent = 1
n_samples_min = int(10**n_samples_min_exponent)
n_samples_max_exponent = 5
n_samples_max = int(10**n_samples_max_exponent)

n_prior = int(max(2e6, n_samples_max))

n_samples_step = int((n_samples_max_exponent-n_samples_min_exponent)*5) + 1
n_fwd_sample_list = torch.logspace(n_samples_min_exponent, n_samples_max_exponent, n_samples_step, dtype=torch.int)


print(n_fwd_sample_list)

T_benchmark = n_samples_max
N_benchmark, M_benchmark = nmc_reuse_partiton(T_benchmark)


scheduler = torch.optim.lr_scheduler.StepLR

scenario_names = ['nmc', 'nmc_reuse', 'dn', 'var_marg', 'var_post', 'nce']

scenario_sample_partition = {
    'nmc':       nmc_partition,
    'nmc_reuse': nmc_reuse_partiton,
    'dn':        lambda x: (x, None),
    'var_marg':  variational_partiton,
    'var_post':  variational_partiton,
    'nce':  variational_partiton,
}

scenario_method = {
    'nmc':       'nmc',
    'nmc_reuse': 'nmc',
    'dn':        'dn',
    'var_marg':  'variational_marginal',
    'var_post':  'variational_posterior',
    'nce':  'nce',
}

var_marg_gradient_steps = 10000

def vm_n_batch_schedule(M, **kwargs):
    return 50
def vm_n_epochs_schedule(M, n_batch, **kwargs):
    return max(min(100, (var_marg_gradient_steps*n_batch)//(M)), 4)
def vm_scheduler_step_size_schedule(n_epochs, **kwargs):
    return n_epochs//3

var_post_gradient_steps = 10000

def vp_n_batch_schedule(M, **kwargs):
    return 50
def vp_n_epochs_schedule(M, n_batch, **kwargs):
    return max(min(200, (var_post_gradient_steps*n_batch)//(M)), 4)
def vp_scheduler_step_size_schedule(n_epochs, **kwargs):
    return n_epochs//3
    
infonce_gradient_steps = 1000

def nce_n_batch_schedule(M, **kwargs):
    return 100
def nce_n_epochs_schedule(M, n_batch, **kwargs):
    return max(min(200, (infonce_gradient_steps*n_batch)//(M)), 4)
def nce_scheduler_step_size_schedule(n_epochs, **kwargs):
    return n_epochs//3

scenario_args = {
    'nmc':       {'reuse_M_samples':False, 'memory_efficient':False},
    'nmc_reuse': {'reuse_M_samples':True , 'memory_efficient':True},
    'dn': {},
    'var_marg': {
        'guide': GMM_guide,
        'guide_kwargs': {'components':10},
        'n_batch': vm_n_batch_schedule,
        'n_epochs': vm_n_epochs_schedule,
        # 'optimizer': torch.optim.Adam,
        'optimizer_kwargs': {'lr': 1e-2},
        'scheduler': torch.optim.lr_scheduler.StepLR,
        'scheduler_kwargs': {'step_size':vm_scheduler_step_size_schedule, 'gamma':0.3},
        'return_guide': False,
        'return_train_loss': False,
        'return_test_loss': True,
        },
    'var_post':{
        'guide': MDN_guide,
     'guide_kwargs': {'components':10, 'hidden_features':[30, 30, 30], 'normalize':True,},
        'n_batch': vp_n_batch_schedule,
        'n_epochs': vp_n_epochs_schedule,
        # 'optimizer': torch.optim.Adam,
        'optimizer_kwargs': {'lr': 1e-4},
        'scheduler': torch.optim.lr_scheduler.StepLR,
        'scheduler_kwargs': {'step_size':vp_scheduler_step_size_schedule, 'gamma':0.3},
        'return_guide': False,
        'return_train_loss': False,
        'return_test_loss': True,
        },
     'nce': {'guide': FullyConnected,
        'K': None,
     'guide_kwargs': {'L': 3, 'H': 20},
        'n_batch': nce_n_batch_schedule,
        'n_epochs': nce_n_epochs_schedule,
        # 'optimizer': torch.optim.Adam,
        'optimizer_kwargs': {'lr': 1e-3},
        'scheduler': torch.optim.lr_scheduler.StepLR,
        'scheduler_kwargs': {'step_size':nce_scheduler_step_size_schedule, 'gamma':0.3},
        'return_guide': False,
        'return_train_loss': False,
        'return_test_loss': True,
     }
}

output_table = [] 

benchmark_seed = 0
torch.manual_seed(benchmark_seed)
prior_samples = prior_dist.sample( (int(n_prior*1.1),) )
benchmark_data_filename = f"data/benchmark/data_lookup_benchmark_{T_benchmark}_{n_design_points}.h5"
# just to be save load prior samples from file as well
prior_samples = generate_lookup_table(benchmark_data_filename, prior_samples, offsets, forward_function, n_parallel=n_parallel)

benchmark_design_dicts = {}
for i, name in enumerate(design_names):
    benchmark_design_dicts[name] = {'index': i, 'offset': offsets[i], 'file': benchmark_data_filename, 'dataset': 'data', 'cost': 1.0,}
benchmark_BED_class = BED_discrete(benchmark_design_dicts, data_likelihood,
                        prior_samples=prior_samples, prior_dist=prior_dist,
                        design2data='lookup_1to1_fast', verbose=False)

for i_bench in range(1, n_benchmarks+1):
    
    torch.manual_seed(i_bench)
    prior_samples = prior_dist.sample( (int(n_prior*1.1),) )
    
    data_filename = f"data/benchmark/data_lookup_{i_bench}_{n_prior}_{n_design_points}.h5"
        
    # just to be save load prior samples from file as well
    prior_samples = generate_lookup_table(data_filename, prior_samples, offsets, forward_function, n_parallel=n_parallel)
    
    design_dicts = {}
    for i, name in enumerate(design_names):
        design_dicts[name] = {'index': i, 'offset': offsets[i], 'file': data_filename, 'dataset': 'data', 'cost': 1.0,}
        
    #while i am at it maybe separate eig from bed class for parallel speed
    BED_class = BED_discrete(design_dicts, data_likelihood,
                        prior_samples=prior_samples, prior_dist=prior_dist,
                        design2data='lookup_1to1_fast', verbose=False)
    
    n_fwd_dict = {}
    n_fwd_info_dict = {}
    
    design_list = []
    t_s_list = []
    
    for T in n_fwd_sample_list:
        
        print(f'\n ===================== benchmark {i_bench} with {T} samples ===================== \n')
        
        optimal_design_dict = {}
        info_dict = {}

        for scenario in scenario_names:

            print(f'              {scenario}                      ')

            filename=f'data/benchmark/avo_methods_{i_bench}_{scenario}_{T}.pkl'

            N, M =  scenario_sample_partition[scenario](T)
        
            s_kwargs = scenario_args[scenario].copy()
            s_kwargs['N'] = N
            if M is not None: s_kwargs['M'] = M
            
            if scenario in ['var_marg', 'var_post'] and M <= 100:
                print(f'Not enough samples for {scenario} with {M} samples')
                continue
                        
            if scenario in ['nmc', 'nmc_reuse'] and T <= 50:
                print(f'Not enough samples for {scenario} with {M} samples')
                continue
            
            if scenario in ['nce'] and (M <= 100 or T > 5000):
                print(f'Not enough samples for {scenario} with {M} samples')
                continue
            
            if scenario in ['var_marg', 'var_post', 'nce']:
                n_batch = s_kwargs['n_batch'](**{'M':M})
                n_epochs = s_kwargs['n_epochs'](**{'M':M, 'n_batch':n_batch})
                step_size = s_kwargs['scheduler_kwargs']['step_size'](**{'n_epochs':n_epochs})
                print(n_batch, n_epochs, step_size)
            
            optimal_design_dict[scenario], info_dict[scenario] = \
                    BED_class.find_optimal_design(
                        design_point_names=design_names,
                        design_budget=design_budget,
                        eig_method=scenario_method[scenario],
                        eig_method_kwargs=s_kwargs,
                        opt_method='iterative_construction',
                        opt_method_kwargs={'random_seed': i_bench},
                        num_workers=n_parallel,
                        filename = filename,
                        )
                    
            n_fwd_dict[T] = optimal_design_dict
            n_fwd_info_dict[T] = info_dict
                        
            design_list.append(optimal_design_dict[scenario])
            t_s_list.append([T, scenario])                
            
    design_list = np.array(design_list)
    
    #calculate benchmarks
    for i_budget in range(1, design_budget+1):
        
        print(f'\n ===================== benchmark {i_bench} with {i_budget} designs ===================== \n')

        filename=f'data/benchmark/avo_benchmark_{i_bench}_{T_benchmark}_{i_budget}.pkl'
                
        benchmark_eig, benchmark_info = \
            benchmark_BED_class.calculate_eig_list(
                design_list=design_list[:, :i_budget],
                method='nmc',
                method_kwargs={'N': N_benchmark, 'M': M_benchmark,
                            'reuse_M_samples':True,
                            'memory_efficient':True},
                filename  =filename,
                num_workers=n_parallel,
                random_seed=benchmark_seed, # to compare all benchmarks should have same seed
            )
        
        bench_counter=0
        for T, scenario in t_s_list:
            
            wall_times = np.array(
                [[n_fwd_info_dict[T][scenario][i_rec+1]['info'][i]['wall_time'] for i in range(n_design_points)] \
                        for i_rec in range(i_budget)])
            
            wall_times = np.sum(wall_times, 0)        
            mean_wall_time = np.mean(wall_times)
            std_wall_time = np.std(wall_times)
            
                    
            table_row = [i_bench, T.item(), scenario, i_budget,
                        benchmark_eig[bench_counter].item(), #benchmark eig
                        mean_wall_time, std_wall_time
                        ]
            output_table.append(table_row)
            
            bench_counter += 1

random_designs_eig_list = []

for i_budget in range(1, design_budget+1):
    
    print(f'\n ===================== random benchmark {n_random_benchmarks} designs ===================== \n')
    
    filename=f'data/benchmark/avo_benchmark_{n_random_benchmarks}_random_designs_{T_benchmark}_{i_budget}.pkl'

    random_designs = [random.sample(design_names, i_budget) for i in range(n_random_benchmarks)]

    heuristic_design_offsets = torch.linspace(500, 1500, i_budget)
    heuristic_design = [str(np.argmin(np.abs(offsets - o)).item()) for o in heuristic_design_offsets]

    random_designs.append(heuristic_design)

    random_benchmark_eig, random_benchmark_info = \
        benchmark_BED_class.calculate_eig_list(
            design_list=random_designs,
            method='nmc',
            method_kwargs={'N': N_benchmark, 'M': M_benchmark,
                        'reuse_M_samples':True,
                        'memory_efficient':True},
            filename  =filename,
            num_workers=n_parallel,
            random_seed=benchmark_seed, # to compare all benchmarks should have same seed
        )
        
    random_designs_eig_list.append(random_benchmark_eig.tolist())    

random_designs_eig_list = np.array(random_designs_eig_list)
np.save(f'data/benchmark/avo_benchmark_random_designs_{n_random_benchmarks}_{T_benchmark}.npy', random_designs_eig_list)
                        
dataframe = pd.DataFrame(output_table,
                        columns=['i_bench', 'T', 'scenario', 'i_budget', 'benchmark_eig', 'mean_wall_time', 'std_wall_time'],
                        index=None)

dataframe.to_csv(f'data/benchmark/avo_benchmark_table_times_{n_benchmarks}.csv')


