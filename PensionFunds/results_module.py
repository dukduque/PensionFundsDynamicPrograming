#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 07:04:44 2019

@author: dduque

Script to process results of each policy
"""
import os
import sys
import pickle


import numpy as np
import PensionFunds.PensionFundDP as pf

def prob_params():
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
    problem_params = T, R, rf, df, I0, c, w, G, w_delta, max_wealth
    return problem_params

def parse_name(file_name):
    f1 = f.replace('.pickle','')
    params = f1.rsplit("_", 3)
    method = params[0]
    p1 = Integer.(params[1])
    p2 = params[2]
    p3 = params[3]
    return method, p1, p2, p3
    

if __name__ == '__main__':
        
    PF_path = '/Users/dduque/MacWorkspace/PorfolioOpt/PensionFunds/'
    '''
    Get problem data
    '''
    problem_params = prob_params()
    T, R, rf, df, I0, c, w, G, w_delta, max_wealth = problem_params
    '''
    Read returns data
    '''
    return_train_file = os.path.join(PF_path, 'returns_train.p' )
    r = pickle.load(open(return_train_file, 'rb'))
    return_test_file = os.path.join(PF_path, 'returns_test.p' )
    simulated_returns = pickle.load(open(return_test_file, 'rb'))
    
    '''
    Read files
    '''
    pf_file = os.listdir(PF_path)
    for f in pf_file:
        if f[-8:]=='0.pickle' and 'power' in f :
            print(f)
            method, p1, p2, p3 = parse_name(f)
            read_out = pickle.load(open('%s%s' %(PF_path,f), 'rb'))
            S, A, F, T,r,w_delta,max_wealth ,simulated_returns, sols_DP = read_out
            methods_dp = sols_DP.keys()
            w_map = pf.w_index(w_delta, max_wealth,len(S))
            try:
                for m in methods_dp:
                    dp_out, DP_sim_results = sols_DP[m]
                    if m != 'Default':
                        V,U = dp_out
                        sim_out = pf.simulation(problem_params,U,w_map,simulated_returns, policy_name=f)
                        pf.plot_policy_and_sim2(T ,S, w_map, U, F, A, G, DP_sim_results, m)
                        pf.plot_policies_comparizon(('Default', sols_DP['Default'][1]),(m, DP_sim_results), G)
            except:
                print(sys.exc_info()[0])
    
#scp dduque@crunch.osl.northwestern.edu:/home/dduque/dduque_projects/PorfolioOpt/PensionFunds/Plots/*.pdf ./PensionFunds/Plots/
#scp dduque@crunch.osl.northwestern.edu:/home/dduque/dduque_projects/PorfolioOpt/PensionFunds/*0.pickle ./PensionFunds/
            