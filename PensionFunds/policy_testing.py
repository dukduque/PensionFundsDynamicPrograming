#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 09:11:39 2019

@author: dduque

Launch a specific version of the DP algorithm
"""

import sys
import os

PF_path = os.path.dirname(os.path.realpath(__file__))
parent_path= os.path.abspath(os.path.join(PF_path, os.pardir))
sys.path.append(parent_path)

import argparse
import pickle
import numpy as np
import PensionFunds.PensionFundDP as pf

def parse_arguments():
    parser = argparse.ArgumentParser("DP laucher")
    parser.add_argument("method", choices=pf.ALG_CHOICES, help='Method to use in the DP')
    parser.add_argument("proc", default=4, type=int, help='Number of processors to use')
    parser.add_argument("-p1" , default=0, type=float, 
                        help='Parameter 1: Value of gamma for power utility and K1 for sigmoidal.'+
                        'For CVAR and SSD this is the value of beta')
    parser.add_argument("-p2" , default=0.5, type=float, 
                        help='Parameter 2: For sigmoidal is the value of k2, for cvar is the value of alpha')
    parser.add_argument("-p3" , default=0, type=float, 
                        help='Parameter 3: For sigmoidal is the value of k3')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    '''
        Algorithm options
    '''
    alg_options = parse_arguments()
    
    '''
        Example parameters
    '''
    T = 40 
    R = 18
    rf = 0.02
    df = 1/(1+rf)
    I0 = 10000
    c = 0.15
    I_star = np.round(I0*((1+rf)**40))
    w = 2/3
    G = np.round(w*I_star*sum(df**k for k in range(1,R+1)))
    
    w_delta = 100
    max_wealth = 3E6
    

    '''
    Output variables to store
    '''
    sols_DP = {}
    
    '''
    Read returns data
    '''
    return_train_file = os.path.join(PF_path, 'returns_train.p' )
    r = pickle.load(open(return_train_file, 'rb'))
    return_test_file = os.path.join(PF_path, 'returns_test.p' )
    simulated_returns = pickle.load(open(return_test_file, 'rb'))
    replicas = len(simulated_returns)
    
    '''
    Wrap problem parameters and dp data
    '''
    problem_params = T, R, rf, df, I0, c, w, G, w_delta, max_wealth
    dp_data = pf.setup(T,r,w_delta,max_wealth)
    
    '''
    Build and simulate default policy
    '''
    default_policy, default_sim = pf.run_default(problem_params, dp_data, r,simulated_returns,plot=False) 
    Y =  np.array([sr[-1] for sr in default_sim]) #Benchmark random variable
    sols_DP['Default'] = (default_policy, default_sim)
    
    
    '''
    Run DP for the specified method
    '''
    method = alg_options.method
    alg_params = (alg_options.p1, alg_options.p2, alg_options.p3)
    policy_name_params = '%s_%.2f_%.2f_%.2f'  %(method,alg_options.p1, alg_options.p2, alg_options.p3)
    
    dp_out = pf.backward_induction_sd_mix_par(problem_params, dp_data, r, Y,default_policy, method=method, method_params = alg_params , method_cond=False, n_threads=alg_options.proc)
    V,U = dp_out
    w_map = dp_data[3]
    DP_sim_results = pf.simulation(problem_params,U,w_map,simulated_returns, policy_name=policy_name_params)
    sols_DP[method] = (dp_out, DP_sim_results)
    
    
    S, A, F, w_map, steps = dp_data
    all_policies_out  = (S, A, F, T,r,w_delta,max_wealth,simulated_returns, sols_DP)
    out_path = os.path.join(PF_path,'%s.pickle' %(policy_name_params))
    pickle.dump(all_policies_out , open(out_path, 'wb'), pickle.HIGHEST_PROTOCOL)
  
    #pf.plot_policy_and_sim2(T ,S, w_map, U, F, A, G, DP_sim_results, method)
    #pf.plot_policies_comparizon(('Default', sols_DP['Default'][1]),(method, DP_sim_results), G)
    
    
    
    
    
    
    
    
    
    
