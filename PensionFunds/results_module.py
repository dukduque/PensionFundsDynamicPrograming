#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 07:04:44 2019

@author: dduque

Script to process results of each policy
"""
import os
import sys

PF_path = os.path.dirname(os.path.realpath(__file__))
parent_path= os.path.abspath(os.path.join(PF_path, os.pardir))
sys.path.append(parent_path)

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
    p1 = float(params[1])
    p2 = float(params[2])
    p3 = float(params[3])
    return method, p1, p2, p3
    

def produce_list():
    f_names = []
#    for g in [4,4.5,5.0,5.5,6.0,6.5,7.0,7.5,7.6,7.8,7.9,8.0,9.0]:
#        f_names.append('%s_%.4f_%.3f_%.2f'  %('utility_power', g, 0.0, 0.0))
#    
#    a=0.99
#    for b in [5,7,8,9,10,11,12,15,20]:
#        for c in [4.0,4.5,5.0,5.3,5.4,5.5,5.6,5.7,6.0]:
#            f_names.append('%s_%.4f_%.3f_%.2f'  %('utility_sigmoidal', a, b, c))
#
#    # CVar using linear penalty and quadratic penalty a is the alpha value and b is beta
#    for b in [0.0,0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5]:
#        for a in {0.05,0.1,0.2,0.3,0.7,0.8,0.9,0.95}:
#            f_names.append('%s_%.4f_%.3f_%.2f'  %('cvar_l', b, a, 0.0))
#            f_names.append('%s_%.4f_%.3f_%.2f'   %('cvar_q', b, a, 0.0))
 
    # SSD
    for b in [0.0,0.001,0.005,0.01,0.05,0.1,0.5]:
        #f_names.append('%s_%.4f_%.3f_%.2f'  %('ssd', b, 0.5, 0.0))
        #f_names.append('%s_%.4f_%.3f_%.2f'  %('ssd_minmax', b, 0.5, 0.0))
        for t in [0.7,0.8,0.9,1.0,1.1,1.2,1.3]:
            f_names.append('%s_%.4f_%.3f_%.2f'  %('ssd_tail_r', b, t, 0.0))
            f_names.append('%s_%.4f_%.3f_%.2f'  %('ssd_tail', b, t, 0.0))
    
    return f_names

def save_results(partial_results):
    assert type(partial_results) == type({})
    with open('pf_results.p', 'wb') as handle:
        pickle.dump(partial_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
def load_results():
    results = None
    with open('pf_results.p', 'rb') as handle:
        results = pickle.load(handle)
    return results

if __name__ == '__main__':
        
    PF_path = '/Users/dduque/MacWorkspace/PorfolioOpt/PensionFunds/'
    PF_path_HD = '/Volumes/DDV_backup/DP_PF_results/'
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
    
    results_summary = load_results()
    
    pf_file = os.listdir(PF_path)
    f_names = produce_list()
    for f in f_names:
        try:
            method, p1, p2, p3 = parse_name(f)
            ins_name = f + '.pickle'
            read_out = pickle.load(open('%s%s' %(PF_path_HD,ins_name), 'rb'))
            S, A, F, T,r,w_delta,max_wealth ,simulated_returns, sols_DP = read_out
            methods_dp = sols_DP.keys()
            w_map = pf.w_index(w_delta, max_wealth,len(S))
        
            for m in methods_dp:
                dp_out, DP_sim_results = sols_DP[m]
                if m != 'Default':
                    V,U = dp_out
                    sim_out, out_stats = pf.simulation(problem_params,U,w_map,simulated_returns, policy_name=f)
                    print(out_stats)
                    results_summary[(method, p1, p2, p3)] = out_stats
                    #save_results(results_summary)
                    import PensionFunds.PensionFundDP as pf
                    pf.plot_policy_and_sim2(T ,S, w_map, U, F, A, G, DP_sim_results, f, True)
                    #pf.plot_policies_comparizon(('default', sols_DP['Default'][1]),(m, DP_sim_results), G, f)
                    sim_out, out_stats = pf.simulation(problem_params, sols_DP['Default'][0],w_map,simulated_returns, policy_name='default')
                    
        except:
            print(sys.exc_info()[0])
        
    for k in results_summary:
        out_str = ''+str(results_summary[k][1:])
        out_str= out_str.replace('(', '')
        out_str= out_str.replace(')', '')
        out_str= out_str.replace(',', '')
        print(k[0],k[1],k[2],k[3],out_str)
    
    
    
    

if False:
    
    #responese = os.system("scp dduque@crunch.osl.northwestern.edu:/home/dduque/dduque_projects/PorfolioOpt/PensionFunds/%s ./PensionFunds/" %(ins_name))
    #command_copy  = "scp %s%s /Volumes/DDV_backup/DP_PF_results/" %(PF_path,ins_name)
    #os.system("rm ./PensionFunds/%s" %(ins_name))
    
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
                        sim_out, out_stats = pf.simulation(problem_params,U,w_map,simulated_returns, policy_name=f)
                        pf.plot_policy_and_sim2(T ,S, w_map, U, F, A, G, DP_sim_results, m)
                        pf.plot_policies_comparizon(('Default', sols_DP['Default'][1]),(m, DP_sim_results), G)
            except:
                print(sys.exc_info()[0])
    
#scp dduque@crunch.osl.northwestern.edu:/home/dduque/dduque_projects/PorfolioOpt/PensionFunds/Plots/*.pdf ./PensionFunds/Plots/
#scp dduque@crunch.osl.northwestern.edu:/home/dduque/dduque_projects/PorfolioOpt/PensionFunds/*0.pickle ./PensionFunds/
#scp dduque@crunch.osl.northwestern.edu:/home/dduque/dduque_projects/PorfolioOpt/PensionFunds/ssd*0.pickle ./PensionFunds/
#scp dduque@crunch.osl.northwestern.edu:/home/dduque/dduque_projects/PorfolioOpt/PensionFunds/ssd_tail*0.pickle /Volumes/DDV_backup/DP_PF_results
