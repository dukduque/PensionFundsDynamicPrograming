#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 00:46:54 2018

@author: dduque
"""
import numpy as np
import pandas as pd
from gurobipy import Model, GRB, quicksum

def load_file(a):
    print(a)

def get_opt_porfolio(data, delta_param, alpha_param):
    r = np.mean(data, axis=0)
    cov = np.cov(data, rowvar=False)
    k = len(r)
    n = len(data)
    m = Model('opt_profolio')
    m.params.OutputFlag = 0
    m.params.NumericFocus = 3
    x = m.addVars(k,lb=0,ub=1,vtype=GRB.CONTINUOUS,name='x')
    norm_p = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='norm')
    p_SD = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='p_var')
    m.update()
    
    sqrt_delta =np.sqrt(delta_param)
    m.addConstr((x.sum()==1), 'portfolio_ctr')
    m.addConstr((quicksum(x[j]*r[j] for j in range(k)) >= alpha_param - sqrt_delta*norm_p ), 'return_ctr' )
    m.addConstr((norm_p*norm_p>=(quicksum(x[j]*x[j] for j in range(k)))), 'norm_def')
    m.addConstr((p_SD*p_SD>=quicksum(cov[i,j]*x[i]*x[j] for i in range(k) for j in range(k))), 'SD_def')
    objfun = p_SD*p_SD + 2*p_SD*sqrt_delta*norm_p + delta_param*norm_p*norm_p
    m.setObjective(objfun, GRB.MINIMIZE)
    
    m.optimize()
    x_sol =np.array([x[j].X for j in range(k)])
    p_mean = r.dot(x_sol)
    p_var  = x_sol.dot(cov.dot(x_sol))
    #print(x_sol, p_mean, p_var)
    #print('norms' , np.linalg.norm(x_sol) , norm_p.X)

    return x_sol, p_mean, p_var

def test_porfolio(portfolio, data_test, steps, replicas):
    assert len(portfolio) == len(data_test[0])
    np.random.seed(0)
    cum_return = np.array([portfolio.copy() for _ in range(replicas)])
    
    for i in range(replicas):
        for t in range(steps):
            rnd_row = np.random.randint(len(data_test))
            return_t = 1+ data_test[rnd_row].squeeze()
            cum_return[i] = cum_return[i]*return_t
    
    #Add porfolito wealth on each replica and get cummulative return
    sim_results = np.sum(cum_return, axis=1)-1
    sim_results.sort()
    sim_results = -1 + (1 + sim_results)**(365/steps)
    cvar_loss = np.average(sim_results[:int(0.05*replicas)])
    
    
    return np.average(sim_results), np.std(sim_results), cvar_loss
    


if __name__ == '__main__':
    
    assets = pd.read_excel('../data/bancolombia/consolidado.xlsx', sheet_name='asset_daily_returns')
    assets = assets.dropna() 
    data_train = np.array(assets.iloc[365:,1:])
    data_test = np.array(assets.iloc[:,1:])
    
    m = len(assets.index)
    target_return = 0.10
    
    sols_delta = {}
    print('%12s %10s %10s %10s' %('delta', 'return',  'SD', 'CVaR_0.05'))
    for exponent in np.arange(-6,2):
        for b in np.arange(1,10.0):
            delta_param = b*(10.0**exponent)
            alpha_param = -1 + (1+target_return)**(1/365)
            x_sol, p_mean, p_var = get_opt_porfolio(data_train,delta_param,alpha_param)
            sols_delta[delta_param] = x_sol
            p_mean_annual = -1 + (1+p_mean)**(365)
            #print('R, SD: %10.4f, %10.4f' %(p_mean_annual,  np.sqrt(m*p_var)))
            
            test_return, test_SD, cvar = test_porfolio(x_sol, data_test, 3*365, 1000)
            print('%12.5e %10.4f %10.4f %10.4f' %(delta_param, test_return,  test_SD, cvar))
            
#            for j in range(len(x_sol)):
#                print('%30s %10.5f' %(assets.columns[j+1],x_sol[j]))
    
    