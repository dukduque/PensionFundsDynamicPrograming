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

def markovitz_dro_wasserstein(data, delta_param, alpha_param, wasserstein_norm=1):
    '''
    Model from Blanchet et al. 2017
    DRO Markovitz reformulation from Wasserstein distance.
    '''
    
    r = np.mean(data, axis=0)
    cov = np.cov(data, rowvar=False)
    k = len(r)
    n = len(data)
    m = Model('opt_profolio')
    m.params.OutputFlag = 0
    m.params.NumericFocus = 3
    x = m.addVars(k,lb=0,ub=1,vtype=GRB.CONTINUOUS,name='x')
    norm_p = m.addVar(lb=0,ub=1, vtype=GRB.CONTINUOUS, name='norm')
    p_SD = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='p_var')
    m.update()
    
    sqrt_delta =np.sqrt(delta_param)
    m.addConstr((x.sum()==1), 'portfolio_ctr')
    m.addConstr((quicksum(x[j]*r[j] for j in range(k)) >= alpha_param - sqrt_delta*norm_p ), 'return_ctr' )
    m.addConstr((p_SD*p_SD>=quicksum(cov[i,j]*x[i]*x[j] for i in range(k) for j in range(k))), 'SD_def')
    objfun = p_SD*p_SD + 2*p_SD*sqrt_delta*norm_p + delta_param*norm_p*norm_p
    m.setObjective(objfun, GRB.MINIMIZE)
    
    if wasserstein_norm == 1:
        regularizer_norm = 'inf'
        m.addConstrs((norm_p>=x[j] for j in range(k)), 'norm_def')
    elif wasserstein_norm ==2 :
        regularizer_norm = 2
        m.addConstr((norm_p*norm_p>=(quicksum(x[j]*x[j] for j in range(k)))), 'norm_def')
    elif wasserstein_norm == 'inf':
        regularizer_norm = 1
        #Note: this works since x>=0
        m.addConstr((norm_p==(quicksum(x[j] for j in range(k)))), 'norm_def')
    else:
        raise 'wasserstain norm should be 1,2, or inf'
    
    
    
    
    
    m.optimize()
    x_sol =np.array([x[j].X for j in range(k)])
    p_mean = r.dot(x_sol)
    p_var  = x_sol.dot(cov.dot(x_sol))
    #print(x_sol, p_mean, p_var)
    #print('norms' , np.linalg.norm(x_sol) , norm_p.X)

    return x_sol, p_mean, p_var


def max_return(r, risk_threshold, cvar_alpha=0.95):
    
    
    r_bar = np.mean(r, axis=0)
    cov = np.cov(r, rowvar=False)    
    n = len(r)
    k = len(r[0])
    m = Model('opt_profolio')
    m.params.OutputFlag = 0
    m.params.NumericFocus = 3
    x = m.addVars(k,lb=0,ub=1, vtype=GRB.CONTINUOUS,name='x')
    z = m.addVars(n,lb=0, vtype=GRB.CONTINUOUS,name='z')
    eta = m.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name='eta')
    m.update()
    
    #Portfolio contraint
    m.addConstr((x.sum()==1), 'portfolio_ctr')
    #Risk constraint
    m.addConstr((eta+(1.0/(n*(1-cvar_alpha)))*z.sum()<=risk_threshold), 'cvar_ctr')
    #CVaR linearlization
    m.addConstrs((z[i]>=quicksum(-r[i,j]*x[j] for j in range(k))-eta  for i in range(n)), 'cvar_linear')
    
    m.setObjective(quicksum(r_bar[j]*x[j] for j in range(k)), GRB.MAXIMIZE)
    m.update()
    
    m.optimize()
    x_sol =np.array([x[j].X for j in range(k)])
    p_mean = r_bar.dot(x_sol)
    p_var  = x_sol.dot(cov.dot(x_sol))
    cvar_loss = (eta+(1.0/(n*(1-cvar_alpha)))*z.sum()).getValue()
    #print(m.status, eta.X, (-1+(1+eta.X)**365), cvar_loss , (-1+(1+cvar_loss)**365))
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
    
    assets = pd.read_excel('../data/bancolombia/Consolidado.xlsx', sheet_name='asset_daily_returns')
    assets = assets.dropna() 
    data_train = np.array(assets.iloc[365:,1:])
    data_test = np.array(assets.iloc[:,1:])
    
    m = len(assets.index)
    target_return = 0.10 #Annual
    alpha_param = -1 + (1+target_return)**(1/365)
    
    
    print('%12s %10s %10s %10s' %('Max CVaR', 'return',  'SD', 'CVaR_0.05'))
    sols_risk_threshold = {}
    for exponent in np.arange(-5,-3):
        for b in np.arange(1,9):
    #y_cvars = [0.1,0.08,0.07,0.06,0.05,0.04,0.03,0.02,0.01,0,-0.01,-0.02,-0.03]
    #for risk_param in [-1+(1+ycvar)**(1/365.0) for ycvar in y_cvars]:
            risk_param = b*(10.0**exponent)
            
            x_sol, p_mean, p_var = max_return(data_train,risk_param)
            sols_risk_threshold[risk_param] = x_sol
            p_mean_annual = -1 + (1+p_mean)**(365)
            #print('R, SD: %10.4f, %10.4f' %(p_mean_annual,  np.sqrt(m*p_var)))
            
            test_return, test_SD, cvar = test_porfolio(x_sol, data_train, 2*365, 1000)
            print('%12.4f %10.4f %10.4f %10.4f' %(-1+(1+risk_param)**365, test_return,  test_SD, cvar))
    
    raise 'stop'
    sols_delta = {}
    print('%12s %10s %10s %10s' %('delta', 'return',  'SD', 'CVaR_0.05'))
    #Solution for delta  = 0 
    x_sol, p_mean, p_var = markovitz_dro_wasserstein(data_train,0,alpha_param,1)
    p_mean_annual = -1 + (1+p_mean)**(365)
    #print('R, SD: %10.4f, %10.4f' %(p_mean_annual,  np.sqrt(m*p_var)))
    sols_delta[0] = x_sol
    test_return, test_SD, cvar = test_porfolio(x_sol, data_test, 365, 1000)
    print('%12.5e %10.4f %10.4f %10.4f' %(0, test_return,  test_SD, cvar))
    for exponent in np.arange(-10,2):
        for b in np.arange(1,10.0):
            delta_param = b*(10.0**exponent)
            
            x_sol, p_mean, p_var = markovitz_dro_wasserstein(data_train,delta_param,alpha_param,1)
            sols_delta[delta_param] = x_sol
            p_mean_annual = -1 + (1+p_mean)**(365)
            #print('R, SD: %10.4f, %10.4f' %(p_mean_annual,  np.sqrt(m*p_var)))
            
            test_return, test_SD, cvar = test_porfolio(x_sol, data_test, 2*365, 1000)
            print('%12.5e %10.4f %10.4f %10.4f' %(delta_param, test_return,  test_SD, cvar))
            
#            for j in range(len(x_sol)):
#                print('%30s %10.5f' %(assets.columns[j+1],x_sol[j]))
    
    