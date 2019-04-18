#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 12:16:05 2018

@author: dduque
"""

#Setup dependencies in this project
import os 
import sys
os.environ['MKL_NUM_THREADS'] = "1" #THIS TELLS NUMPY TO DO NOT MULTITHREAD
PF_path = os.path.dirname(os.path.realpath(__file__))
parent_path= os.path.abspath(os.path.join(PF_path, os.pardir))
sys.path.append(parent_path)

import numpy as np
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.colors as col
from matplotlib import colors 
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.mlab as mlab
from matplotlib import rc
#Set up font for latex
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
import scipy.stats
import scipy.stats as st
import pandas as pd
import pickle
from PensionFunds.NORTA import fit_NORTA, NORTA, build_empirical_inverse_cdf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.distributions.empirical_distribution import ECDF
from fitter import Fitter
import multiprocessing as mp
from itertools import product
from time import time

ALG_UTILITY_POWER = 'utility_power'
ALG_UTILITY_SIGMO = 'utility_sigmoidal'
ALG_CVAR_PENALTY_LIN = 'cvar_l'
ALG_CVAR_PENALTY_QUA = 'cvar_q'
ALG_SSD = 'ssd'
ALG_SSD_TAIL= 'ssd_tail'
ALG_SSD_MINMAX='ssd_minmax'

ALG_CHOICES = [ALG_UTILITY_POWER,ALG_UTILITY_SIGMO,ALG_CVAR_PENALTY_LIN,ALG_CVAR_PENALTY_QUA,ALG_SSD,ALG_SSD_TAIL,ALG_SSD_MINMAX]

def utility_function(x, G, power, *args):
    '''
    params:
        x (float): wealth
        G (float): target wealth
    '''
    if power:
        u_gamma = args[0]
        return (1/(1-u_gamma))*(x/G)**(1-u_gamma)
    else:    
        c1 = args[0] #101
        c2 = args[1] #100
        c3 = args[2] #-0.1
        return 1/(c1+np.exp(-c2*(x/G)+c3))
 
def w_index(w_delta, max_wealth, steps):
    def f(x):
        p = np.floor(x/w_delta) # np.round(x,0)/w_delta
        p = np.minimum(p, steps-1)
        p = np.maximum(p, 0)
        p = p.astype(int)
        return p
    return f 

def setup(T, r, w_delta=100, max_wealth=2E6):
    steps = int(max_wealth/w_delta)
    F = r.keys()
    F = list(F)
    F.sort()
    A = []  
    for (i,ai) in enumerate(F):
        ac = np.zeros(len(F))
        ac[i] = 1
        A.append(ac)
        for (j,aj) in enumerate(F):
            if j>i:
                for alp in [0.05*i for i in range(1,20)]:
                    ac = np.zeros(len(F))
                    ac[i] = 1-alp
                    ac[j] = alp
                    A.append(ac)
    A = np.array(A)
    w_map = w_index(w_delta, max_wealth,steps)
    S = np.array([i*w_delta for i in range(steps) if i*w_delta<=max_wealth])
    return S, A, F, w_map, steps

def backward_induction_sd_mix_par(problem_data, dp_data, r , Y=None, Y_policy=None, method=None, method_params = None,  method_cond=False, n_threads=1):
    '''
    Runs backward induction to find the optimal policy of a dynamic
    asset allocation problem. This method implements a parralle version
    of the DP.
    
    Args:
        T (int): number of time periods in the planning horizon
        G (float): a target value used in the model with utility function
        rf (float): risk free rate
        r (dict): historical returns of the funds
        I (float): initial salary
        c (float): fraction of the anual salary contributed to retirment
        Y (ndarray): realizations of the benchmark
        w_delta (float): discretization gap used in in the DP for the state variable
        method (str): method used in the optimization
        n_threads (int): number of threads to used in the DP
    '''
    print('Runing DP with %s and %i threads' %(method, n_threads))
    
    tnow = time()
    T, R, rf, df, I0, c, w, G, w_delta, max_wealth = problem_data
    S, A, F, w_map, steps = dp_data
    
    V = np.zeros((T+1,steps))
    U = {}
    r_mat = np.zeros((len(F),len(r[F[0]])))
    for (i,ai) in enumerate(F):
        r_mat[i,:]= r[ai]
    
    act_ret = A.dot(r_mat)
    
    #R = np.eye(steps) #PMF of final wealth
    beta_cvx, var_val, cvarY, Yq,  Ytail, SDD_constant = None, None, None, None, None, None
    if type(Y)!=type(None) and method in [ALG_SSD,ALG_SSD_TAIL, ALG_SSD_MINMAX, ALG_CVAR_PENALTY_LIN, ALG_CVAR_PENALTY_QUA]:
        beta_cvx = method_params[0]
        var_val = method_params[1]
        cvarY = {}
        cvarY_unconditioned = cvar(-Y,var_val)  
        for (i,s) in enumerate(S):
            if method_cond:
                newY =  s*(1+Y_policy[T-1,i].dot(r_mat))+c*(I0*(1+rf)**(T-1))
                cvarY[s] = cvar(-newY,var_val)  
            else:
                cvarY[s] = cvarY_unconditioned
        #cvarY = cvar(-Y,var_val) 
        if method_cond == False:
            Yq = np.percentile(Y,q=[i for i in range(0,101)],axis=0)
            Ytail= Yq[Yq<=G]
            YY = np.tile(Yq, (len(Yq), 1))
            SSD =  np.maximum(Yq - YY.transpose(), 0 )
            SDD_constant = SSD.mean(0)
            if method == ALG_SSD_TAIL:
                SDD_constant = SDD_constant[Yq<=G]
    
    print('Method parameters: ' , method_params)

    for (i,s) in enumerate(S):
        assert w_map(s)==i, "NUMS %i %i %i" %(i,s,w_map(s))
    
    # Value in the last stage
    if method == ALG_UTILITY_POWER:
        V[T,:] = utility_function(S, G, True, *method_params ) 
    if method == ALG_UTILITY_SIGMO:
        V[T,:] = utility_function(S, G, False, *method_params) 
    else:
        V[T,:] = S# utility_function(S,G) 
    
    print("Number of processes: ", n_threads)
    p  = mp.Pool(processes=n_threads)

    for t in np.arange(T-1, -1, -1):
        I_t = I0*(1+rf)**t
        print('solving ', t, ' ' , I_t)
        #Launch parallelization for each possible state
        par_data = product(S, [(w_delta, max_wealth,steps)],[act_ret],[c],[I_t],[method],[Ytail], [Yq], [t],[T],[V[t+1,:]],[A],[SDD_constant],[cvarY],[var_val], [Y_policy[T-1,0]], [r_mat], [beta_cvx])
        out_par = p.map(dp_parallel,par_data)
        #out_par = p.map(fake_parralel,S)
        V[t,:] = np.array([par_res[0] for par_res in out_par])
        for (i,s) in enumerate(S):
            U[t,i] = out_par[i][1]
    p.terminate()
    total_time = time()-tnow
    print('CPU time: %10.2f' %(total_time) )
    return V,U

def fake_parralel(x):
    max_n = int(np.log(x+10)) + 10000
    out_val = 0
    A = np.random.normal(size=(190,5))
    r_mat =np.random.normal(size=(5,1000))
    re = A.dot(r_mat)
    re2  = x*(1+re) + max_n*0.4 
    re3 = 10#np.sum(re2)/10000
    for i in range(max_n):
        out_val += i
    return out_val, max_n
        
        
def dp_parallel(dp_data):
    
    '''
    ================================
    Parse all input data
    '''
    s = dp_data[0]
    w_map = w_index(*dp_data[1])
    act_ret = dp_data[2]
    c = dp_data[3]
    I_t = dp_data[4]
    method = dp_data[5]
    Ytail = dp_data[6]
    Yq = dp_data[7]
    t = dp_data[8]
    T = dp_data[9]
    Vt1 = dp_data[10]
    Actions = dp_data[11]
    SDD_constant = dp_data[12]
    cvarY = dp_data[13]
    var_val = dp_data[14]
    Y_policy = dp_data[15] #Static policy of the benchmark
    r_mat = dp_data[16]
    beta_cvx = dp_data[17]
    '''
    ================================
    Solve DP for state s and time t
    '''
    s_index = w_map(s)
    arg_max = None
    V_s = -np.inf
#    X = s*(1+act_ret)+c*I_t
#    Xind = w_map(X)
    
#    if t >= T-1 and method in [ALG_SSD,ALG_SSD_TAIL, ALG_SSD_MINMAX] and type(Yq)==type(None):
#        newY =  s*(1+Y_policy.dot(r_mat))+c*I_t
#        Yq = np.percentile(newY,q=[i for i in range(0,101)],axis=0)
#        Ytail = Yq[Yq<=G]
#        YY = np.tile(Yq, (len(Yq), 1))
#        SSD =  np.maximum(Yq - YY.transpose(), 0)
#        SDD_constant = SSD.mean(0)
#        print('Tamano SSD Y ' , len(SDD_constant))
#   
#    for (k,a) in enumerate(Actions):
#        v_a = None
#        s_a_i = Xind[k]
#        if t >= T-1:
#            exp_v = (1/len(s_a_i))*np.sum(Vt1[s_a_i]) #Expectation
#            if method == ALG_SSD:
#                XX = np.tile(X[k], (len(Yq), 1))
#                SSD =  np.maximum(Yq - XX.transpose(), 0)
#                SDD_mean = SSD.mean(0)
#                SSD_violations =  np.maximum(0, SDD_mean - SDD_constant)
#                v_a = beta_cvx * exp_v - (1-beta_cvx) * SSD_violations.sum()
#            elif method == ALG_SSD_TAIL:
#                assert len(Ytail)==len(SDD_constant), 'SSD constant vector has a different dimension'
#                XX = np.tile(X[k], (len(Ytail), 1))
#                SSD =  np.maximum(Ytail - XX.transpose(), 0)
#                SDD_mean = SSD.mean(0)
#                SSD_violations =  np.maximum(0, SDD_mean - SDD_constant)
#                v_a = beta_cvx * exp_v - (1-beta_cvx) * SSD_violations.sum()
#            elif method == ALG_SSD_MINMAX:
#                XX = np.tile(X[k], (len(Yq), 1))
#                SSD =  np.maximum(Yq - XX.transpose(), 0)
#                SDD_mean = SSD.mean(0)
#                v_a = beta_cvx * exp_v - (1-beta_cvx) * np.max(SDD_mean - SDD_constant)
#            elif method == ALG_CVAR_PENALTY_LIN:
#                cvarX = cvar(-X[k],var_val)
#                v_a = beta_cvx * exp_v - (1-beta_cvx) * np.maximum(0,cvarX-cvarY[s])
#                #if cvarX > cvarY[s]:
#                #    v_a = v_a  - 100*(cvarX-cvarY[s])
#            elif method == ALG_CVAR_PENALTY_QUA:
#                cvarX = cvar(-X[k],var_val)
#                v_a = beta_cvx * exp_v - (1-beta_cvx) * np.power(np.maximum(0,cvarX-cvarY[s]),2)
#                #if cvarX > cvarY[s]:
#                #    v_a = v_a  - (cvarX-cvarY[s])**2
#            elif method == ALG_UTILITY_POWER or method == ALG_UTILITY_SIGMO:
#                v_a = exp_v
#            else:
#                raise 'unimplemented method'
#        else:
#            v_a = (1/len(s_a_i))*np.sum(Vt1[s_a_i]) #Expectation
#        if v_a>=V_s: #Choose less risk of alternative optima
#            V_s = v_a
#            arg_max = a
#    return V_s, arg_max 
    return fake_parralel(s)



def cvar_plus(x, alp):
    '''
    Computes the cvar of a given vector of obsebations
    at a particular quantile value
    Note: apl should be > 0.5 (e.g., 0.95) as this formula looks at
          the upper tail an x_in is looked as loses
    Args:
        x(ndarray): vector with realizations
        alp (float): quantile at risk (0,1)
    '''
    var_alp=np.percentile(x, q = int(alp*100))
    return x[x>=var_alp].mean()

def cvar(x_in, alp, p=None):
    '''
    Computes the cvar of a given vector of obsebations
    at a particular quantile value.
    Proposition 8 Rockafellar and  Uryasev (2002)
    https://www.ise.ufl.edu/uryasev/files/2011/11/cvar2_jbf.pdf
    
    Note: apl should be > 0.5 (e.g., 0.95) as this formula looks at
          the upper tail an x_in is looked as loses
    Args:
        x_in(ndarray): vector with realizations
        alp (float): quantile at risk (0,1)
        p (ndarray): Optional version of probabilities
            Assumes the vector x is sorted
    '''
    
    assert alp > 0, 'Value of alpha needs to be in (0,1) '
    x = np.sort(x_in)
    n = len(x)
    
    if type(p)==type(None):
        p =  np.ones_like(x_in)/len(x_in)
   
    #Compute Var
    k_alpha = int(n*alp)  + 1
    while (not p[:k_alpha+1].sum() >= alp > p[:k_alpha].sum()) and (k_alpha>0):
        k_alpha = k_alpha - 1
        
    assert p[:k_alpha+1].sum() >= alp > p[:k_alpha].sum()
    var_alpha = x[k_alpha]
    
    cvar1 = (p[:k_alpha+1].sum() - alp)*var_alpha
    cvar2 = np.sum(p[k_alpha+1:]*x[k_alpha+1:])
    _cvar = (1/(1-alp))*(cvar1+cvar2)

    return _cvar

def check_SSD(X,Y):
    '''
    Checks the fraction of constraints that satisfy that
    X dominates Y in the second order stochastic.
    '''
    ny = len(Y)
    n_ctr = 0 
    for i in range(ny):
        Yi = Y[i]
        EXmax = np.mean(np.maximum(Yi - X, 0))
        EYmax = np.mean(np.maximum(Yi - Y, 0))
        if EXmax <= EYmax:
            n_ctr +=1
    
    return n_ctr/ny
    
def create_default_policy(T,w_delta,max_wealth):
    '''
    Creates the defaul policy for chilean system. This policy uses funds
    B, C, and D  transicioning from B to C after 14 years and from C to D
    after 34 years.
    Args:
        T (int): number of time periods
    Return:
        U (dict): dictionary with the policy for every time period and state. 
    '''
    steps = int(max_wealth/w_delta) + 2
    S = np.array([i*w_delta for i in range(steps) if i*w_delta<=max_wealth])
    U = {}
    for t in range(T):
        for s in range(len(S)):
            if t <=14:
                U[t,s] = np.array([0,1.0,0,0,0])
            elif t <=15:
                U[t,s] = np.array([0,0.8,0.2,0,0])
            elif t <=16:
                U[t,s] = np.array([0,0.6,0.4,0,0])
            elif t <=17:
                U[t,s] = np.array([0,0.4,0.6,0,0])
            elif t <=18:
                U[t,s] = np.array([0,0.2,0.8,0,0])
            elif t <=34:
                U[t,s] = np.array([0,0,1.0,0,0])
            elif t <=35:
                U[t,s] = np.array([0,0,0.8,0.2,0])
            elif t <=36:
                U[t,s] = np.array([0,0,0.6,0.4,0])
            elif t <=37:
                U[t,s] = np.array([0,0,0.4,0.6,0])
            elif t <=38:
                U[t,s] = np.array([0,0,0.2,0.8,0])
            else:
                U[t,s] = np.array([0,0,0,1.0,0])
    return U
            
        
            #def_pol = {(t,w_map(s)):('B' if t<=14 else ('C' if 14<t<=34 else 'D')) for t in range(T) for s in S}
def simulation(setup_params, U, w_map, r_sim, fix_policy=None, policy_name =""):
    np.random.seed(0)
    T, R, rf, df, I0, c, w, G, w_delta, max_wealth = setup_params 
    I = [I0*(1+rf)**t for t in range(T)]
    replications = len(r_sim)
    wealth_sims = []
    for k in range(replications):
        wealth_k = [0]
        for t in range(T):
            policy = U[t,w_map(wealth_k[t])] if fix_policy==None else fix_policy
            wealth_k.append((1+r_sim[k,t,:].dot(policy))*wealth_k[t] + c*I[t])
            
        wealth_sims.append(wealth_k)
    if fix_policy == None:
        fix_policy = 'DP'
    wealths = np.array([sr[-1] for sr in wealth_sims])
    unmet_fraction = wealths[wealths <= G].size / wealths.size
    p70 = 1- wealths[wealths <= 0.7*G].size / wealths.size
    p80 = 1- wealths[wealths <= 0.8*G].size / wealths.size
    p90 = 1- wealths[wealths <= 0.9*G].size / wealths.size
    p95 = 1- wealths[wealths <= 0.95*G].size / wealths.size
    p100 = 1- wealths[wealths <= G].size / wealths.size
    mu = np.mean(wealths)
    below_mu = wealths[wealths<=mu] - mu
    above_mu = wealths[wealths>mu] - mu
    sd_m = np.sqrt(np.sum(below_mu**2)/len(wealths))
    sd_p = np.sqrt(np.sum(above_mu**2)/len(wealths))
    exp_short = np.mean(np.maximum(0,G - wealths))#(G-np.mean(wealths[wealths <= G]))
   
    print('%15s %10s %10s %10s %10s %10s %10s %10s %10s %10s' %('Policy', 'Mean', 'SD-', 'SD+', '70%', '80%', '90%', '95%', '100%' ,  'E.Shortfal' ))
    print('%15s %10.2f %10.2f %10.2f %10.3f %10.3f %10.3f %10.3f %10.3f %10.2f' %(policy_name, mu, sd_m, sd_p, p70, p80, p90, p95, p100, exp_short))
    
    return wealth_sims

def gen_yearly_returns(Funds, monthly_returns, n_years):
    '''
    n_years(int): Number of returns to generate
    '''
    np.random.seed(0)
    n_months = len(monthly_returns[Funds[0]])
    yearly_returns = {f:[] for f in Funds}
    for i in range(n_years):
        initial_month = np.random.randint(n_months) 
        months = list(range(initial_month, initial_month+12))
        if initial_month > n_months - 12:
            for k in range(12):
                if months[k] >= n_months:
                    months[k] -= n_months
        for a in Funds:
            r_a = 1+monthly_returns[a][months]
            y_r_a = np.product(r_a) - 1
            yearly_returns[a].append(y_r_a)
    for a in Funds:   
        yearly_returns[a] = np.array(yearly_returns[a])
    return yearly_returns

def plot_simulation(sim_results, policy, style): 
    if style ==0 :
        fig, ax = plt.subplots()
        cols = cm.rainbow(np.linspace(0, 1, len(sim_results)))
        for (k,result) in enumerate(sim_results):
            ax.plot(result, color=cols[k] )
        ax.plot([G for _ in range(0,T+2)] , color='blue')
        plt.tight_layout()
        plt.show()
    else:
        wealths = [sr[-1] for sr in sim_results]
        wealths.sort()
        bins = 100
        fig, ax = plt.subplots()
        heights,bins = np.histogram(wealths,bins=bins)
        heights = heights/sum(heights)
        ax.bar(bins[:-1],heights,width=(max(bins) - min(bins))/len(bins), color="red", alpha=0.6)
        max_height = np.max(heights)
        ax.set_ylim(0,max_height)
        ax.set_xlim(wealths[0],wealths[-2])
        ax.set_yticks(np.linspace(0, max_height, 5))
        ax.set_xlabel('Wealth at T')
        ax.set_ylabel('Frequency')
        axR = ax.twinx()
        axR.set_xlim(wealths[0],wealths[-2])
        axR.hist(wealths[:], bins=bins, normed=True, cumulative=True, label='CDF', histtype='step', alpha=0.8, color='b')
        #ecdf = ECDF(wealths)
        #axR.plot(ecdf.x,ecdf.y)
        axR.set_yticks(np.linspace(0, 1, 5))
        axR.set_ylim(0,1)
        axR.set_ylabel('Cumulative')
        ax.grid(True)
        ax.axvline(G)
        plt.tight_layout()
        plt.show()

def plot_policies_comparizon(p1_results, p2_results,G):
    wealths = [sr[-1] for sr in p1_results[1]]
    wealths.sort()
    bins = 300
    heights,bins = np.histogram(wealths,bins=bins)
    n_pdf = 10000
    fig, ax = plt.subplots()
    hist_dist1 = scipy.stats.rv_histogram((heights,bins))
    X = np.linspace(wealths[0], wealths[-1], n_pdf)
    max_p1 = np.max(hist_dist1.pdf(X))
    ax.fill_between(X,hist_dist1.pdf(X),np.zeros_like(X),  alpha=0.5)
    
    axR = ax.twinx()
    axR.plot(X, hist_dist1.cdf(X), label=p1_results[0])
    
    
    wealths = [sr[-1] for sr in p2_results[1]]
    wealths.sort()
    bins = 300
    heights,bins = np.histogram(wealths,bins=bins)
    max_p2 = 0
    if len(wealths)>0:
        hist_dist2 = scipy.stats.rv_histogram((heights,bins))
        X = np.linspace(wealths[0], wealths[-1], n_pdf)
        max_p2 = np.max(hist_dist2.pdf(X))
        ax.fill_between(X,hist_dist2.pdf(X),np.zeros_like(X), alpha=0.5, color='red')
        axR.plot(X, hist_dist2.cdf(X), label=p2_results[0], color='red')
    
    
    ax.set_yticks(np.linspace(0, np.maximum(max_p1,max_p2), 11))
    ax.set_ylim(0,np.maximum(max_p1,max_p2))
    ax.set_xlabel('Wealth at T')
    ax.set_ylabel('Frequency')
    ax.set_yticklabels(['%6.2e' %(num) for num in np.linspace(0, np.maximum(max_p1,max_p2), 11)])
    ax.grid('on')
    axR.set_yticks(np.linspace(0, 1, 11))
    axR.set_ylim(0,1)
    axR.set_ylabel('Cumulative')
    axR.axvline(G , label='G', color='black')
    axR.legend(loc='best', shadow=True, fontsize='small')
    plt.tight_layout()
    plt.show()

def plot_policy(T, S, w_map, policy, Funds, G):
    F = {a:i for (i,a) in enumerate(Funds)}
    U_plot = np.array([[F[policy[t,w_map(s)]] for t in range(T)] for s in S if s<=G*1.5])
    U_plot[w_map(G)-6:w_map(G)+6,:] = -1 #Draw G
    cmap = colors.ListedColormap(['black', 'red', 'blue','green', 'orange', 'yellow' ])
    fig, ax = plt.subplots()
    steps_displayed = len(U_plot)
    im = ax.imshow(U_plot, cmap=cmap,  interpolation='none', aspect=T/steps_displayed ,origin='lower', vmin=-1, vmax=len(Funds)-1)
    
    # We want to show all ticks...
    ax.set_xticks(np.arange(0,T+1,5))
    y_ticks = np.arange(0,steps_displayed,int(steps_displayed/10))
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(S[y_ticks])
    # colormap used by imshow
    values = [F[a] for a in Funds]
    cols = [ im.cmap(im.norm(value)) for value in values]
    patches = [ mpatches.Patch(color=cols[i], label="Fund {l}".format(l=Funds[values[i]]) ) for i in range(len(values)) ]
    ax.legend(handles=patches, bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0. )
    ax.set_xlabel('Years')
    ax.set_ylabel('Wealth ($USD)')
    plt.tight_layout()
    plt.show()
    
def plot_policy_and_sim(T, S, w_map, policy, Funds, G, sim_results, plot_name):
    F = {a:i for (i,a) in enumerate(Funds)}
    U_plot = np.array([[F[policy[t,w_map(s)]] for t in range(T)] for s in S if s<=G*1.5])
    U_plot[w_map(G)-10:w_map(G)+10,:] = -1 #Draw G
    #Draw simulation 5-95%
    U_plot2 = np.copy(U_plot)
    for t in range(T):
        r_t = np.array([sim[t] for sim in sim_results])
        r_t.sort()
        w_5 = r_t[int(len(r_t)*0.05)]
        w_95 = r_t[int(len(r_t)*0.95)]
        U_plot2[w_map(w_5):w_map(w_95),t] = -1
        #U_plot2[w_map(w_95)-10:w_map(w_95),t] = -1
        

    cmap = colors.ListedColormap(['black', 'red', 'blue','green', 'orange', 'yellow' ])
    fig, ax = plt.subplots()
    steps_displayed = len(U_plot)
    im = ax.imshow(U_plot, cmap=cmap,  interpolation='none', aspect=T/steps_displayed ,origin='lower', vmin=-1, vmax=len(Funds)-1)
    im2 = ax.imshow(U_plot2, cmap=cmap,  interpolation='none', aspect=T/steps_displayed ,origin='lower', vmin=-1, vmax=len(Funds)-1, alpha=0.3)
    
    # We want to show all ticks...
    ax.set_xticks(np.arange(0,T+1,5))
    y_ticks = np.arange(0,steps_displayed,int(steps_displayed/10))
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(S[y_ticks])
    # colormap used by imshow
    values = [F[a] for a in Funds]
    cols = [ im.cmap(im.norm(value)) for value in values]
    patches = [ mpatches.Patch(color=cols[i], label="Fund {l}".format(l=Funds[values[i]]) ) for i in range(len(values)) ]
    ax.legend(handles=patches, bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0. )
    ax.set_xlabel('Years')
    ax.set_ylabel('Wealth ($USD)')
    plt.tight_layout()
    plt.show()


def plot_policy_and_sim2(T, S, w_map, policy, funds ,actions, G, sim_results,plot_name):
    
    #basic_cols = ['red', 'yellow','cyan', 'lime', 'blue']
    basic_cols = ['red', 'blue', 'lime', 'magenta', 'yellow']
    basic_cols_rgs = np.array([col.to_rgb(c) for c in basic_cols])
    policy_colors = actions.dot(basic_cols_rgs)
    policy_colors = np.vstack((col.to_rgb('black') , policy_colors))
    actions_str = [str(pc) for pc in actions]
    F = {a:i for (i,a) in enumerate(actions_str)}
    U_plot = np.array([[F[str(policy[t,w_map(s)])] for t in range(T)] for s in S if s<=G*1.5])
    U_plot[w_map(G)-10:w_map(G)+10,:] = -1 #Draw G
    #Draw simulation 5-95%
    U_plot2 = np.copy(U_plot)
    for t in range(T):
        r_t = np.array([sim[t] for sim in sim_results])
        r_t.sort()
        w_5 = r_t[int(len(r_t)*0.05)]
        w_95 = r_t[int(len(r_t)*0.95)]
        #U_plot2[w_map(w_5):w_map(w_95),t] = -1
        U_plot2[w_map(w_5):w_map(w_5)+10,t] = -1
        U_plot2[w_map(w_95)-10:w_map(w_95),t] = -1
        

    cmap = colors.ListedColormap(policy_colors)
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
    steps_displayed = len(U_plot)
    im = ax.imshow(U_plot, cmap=cmap,  interpolation='none', aspect=T/steps_displayed ,origin='lower', vmin=-1, vmax=len(actions)-1)
    im2 = ax.imshow(U_plot2, cmap=cmap,  interpolation='none', aspect=T/steps_displayed ,origin='lower', vmin=-1, vmax=len(actions)-1, alpha=0.8)
    
    # We want to show all ticks...
    ax.set_xticks(np.arange(0,T+1,5))
    y_ticks = np.arange(0,steps_displayed,int(steps_displayed/10))
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(S[y_ticks])
    # colormap used by imshow
    #values = [F[a] for a in funds]
    cols = basic_cols#[ im.cmap(im.norm(value)) for value in values]
    patches = [ mpatches.Patch(color=cols[i], label="Fund {l}".format(l=funds[i]) ) for i in range(len(funds)) ]
    #if plot_name == 'Default':
    ax.legend(handles=patches, bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.5 )
    #fig.colorbar(im)
    ax.set_xlabel('Years')
    ax.set_ylabel('Wealth (\$USD)')
    plt.tight_layout() 
    plot_file = os.path.join(PF_path,'Plots/%s.pdf' %(plot_name) )
    #plt.savefig(plot_file, bbox_inches = 'tight', pad_inches = 1)
    pp = PdfPages(plot_file)
    pp.savefig(fig)
    pp.close()
    plt.show()
    

# scp dduque@crunch.osl.northwestern.edu:/home/dduque/dduque_projects/PorfolioOpt/PensionFunds/Plots/*.pdf ./PensionFunds/Plots/

    #Plot of the leyend
    L_plot = []
    pairs = [(i,j) for (i,f1) in enumerate(funds) for (j,f2) in enumerate(funds) if i<j]
    for alp in [0.05*i for i in range(0,21)]:
        L_plot.append([])
        for (i,j) in pairs:  
            ac = np.zeros(len(funds))
            ac[i] = 1-alp
            ac[j] = alp
            ac_map = F[str(ac)]
            L_plot[-1].append(ac_map)
    L_plot = np.array(L_plot)
    cmap = colors.ListedColormap(actions.dot(basic_cols_rgs))
    fig, ax = plt.subplots()
    steps_displayed = len(U_plot)
    im = ax.imshow(L_plot, cmap=cmap,  interpolation='none', aspect='auto' ,origin='lower', vmin=0, vmax=len(actions)-1)
    ax.set_xticks(np.arange(0,len(pairs),1))
    ax.set_xticklabels(["%s-%s" %(funds[p[0]],funds[p[1]]) for p in pairs])
    ax.set_yticks(np.arange(0,21,1))
    ax.set_yticklabels(np.arange(0,21,1)/20)
    plt.tight_layout()
    plt.show()
    
#==============================================================================
# #    show_steps = int(steps/2)
#==============================================================================
#    X = {}
#    Y = {}
#    for a in Funds:
#        X[a]= [t for t in range(T) for i in range(show_steps) if policy[t,i]==a] 
#        Y[a] = [S[i] for t in range(T) for i in range(show_steps) if policy[t,i]==a]
#   
#    fig, ax = plt.subplots()
#    fund_col = {'a':'purple', 'b':'red', 'c':'white', 'd':'blue', 'e':'green'}
#    for a in Funds:
#        ax.scatter(X[a], Y[a], color = fund_col[a], marker='s', label=a)
#    ax.legend(loc='best', shadow=True, fontsize='small')
#    ax.plot([G for _ in range(1,T+1)])
#    ax.set_xlabel('Years')
#    ax.set_ylabel('Wealth')
#    plt.tight_layout()
#    plt.show()

def get_investment_profiles(sim_result, policy, w_map, T, A, funds,  quantiles=[i*5 for i in range(1,21)]):
    
    wealths = np.array(sim_result)
    wealth_profiles = np.percentile(wealths,q=quantiles,axis=0)
    for (i,q) in enumerate(quantiles):
        #print('Profile ', q)
        w_prof = wealth_profiles[i]
        states_id = w_map(w_prof)
        for t in range(T):
            pass #print('%i  %10.0f' %(t, w_prof[t]) , '  ', policy[t,states_id[t]])
    
    basic_cols = ['red', 'blue', 'lime', 'magenta', 'yellow']
    basic_cols_rgs = np.array([col.to_rgb(c) for c in basic_cols])
    policy_colors = A.dot(basic_cols_rgs)
    actions_str = [str(pc) for pc in A]
    F = {a:i for (i,a) in enumerate(actions_str)}
    U_plot = np.array([[F[str(policy[t,w_map(wealth_profiles[i,t])])] for t in range(T)] for (i,q) in enumerate(quantiles)])
    cmap = colors.ListedColormap(policy_colors)
    fig, ax = plt.subplots()
    steps_displayed = len(U_plot)
    im = ax.imshow(U_plot, cmap=cmap,  interpolation='none', aspect=T/steps_displayed ,origin='lower', vmin=0, vmax=len(A)-1)
     # We want to show all ticks...
    ax.set_xticks(np.arange(0,T+1,5))
    y_ticks = np.arange(0,len(quantiles),1)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(quantiles)
    # colormap used by imshow
    #values = [F[a] for a in funds]
    cols = basic_cols#[ im.cmap(im.norm(value)) for value in values]
    patches = [ mpatches.Patch(color=cols[i], label="Fund {l}".format(l=funds[i]) ) for i in range(len(funds)) ]
    ax.legend(handles=patches, bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0. )
    ax.set_xlabel('Years')
    ax.set_ylabel('Simulation Quantiles')
    plt.tight_layout()
    plt.show()

   
def create_marginal_fun(best_dist, best_params):
    dist = getattr(st, best_dist)
    def marginal_inv(q):
        print(dist)
        rv = dist(*best_params)
        print(rv.stats('mvsk'))
        return dist.ppf(q,*best_params)
    return marginal_inv
    
def fit_returns(data):
    '''
    Peforms a goodness of fit test for each fund
    Args:
        data(ndarray):
    '''
    marginals = [] 
    dist_names = []
    candidates =['norm','gamma', 'dgamma',   'johnsonub', 'lognorm', 't', 'weibull_max', 'weibull_min']
    #candidates =['johnsonsb', 'johnsonub', 'lognorm','t']
    for j in range(len(data[0])):
        dj = data[:,j]
        bin_num = int(np.round((dj.max()-dj.min())/(2*st.iqr(dj)*(len(dj)**(-1/3)))))
        f = Fitter(dj, distributions=candidates, bins=bin_num, verbose=False)
        f.fit()
        best_dist = f.df_errors.sumsquare_error.idxmin()
        print(best_dist)
        best_parmas = f.fitted_param[best_dist]
        marginals.append(create_marginal_fun(best_dist, best_parmas))
    for fm in marginals:
        print(fm(0.5))
   
    return marginals, dist_names
  

def run_default(setup_params, dp_data, r, sim_returns, plot=False): 
    T, R, rf, df, I0, c, w, G, w_delta, max_wealth = setup_params  
    S, A, F, w_map, steps  = dp_data
    def_pol = create_default_policy(T,w_delta,max_wealth)
    Default_sim_results = simulation(setup_params,def_pol,w_map,sim_returns, policy_name="%10s" %("Default"))
    #plot_simulation(Default_sim_results, def_pol, style=1)
    if plot:
        plot_policy_and_sim2(T ,S, w_map,  def_pol, Funds, A, G, Default_sim_results, 'Default')
    return def_pol, Default_sim_results
    
if __name__ == '__main__':
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
    max_wealth = 8E5
   
    case_params = T, R, rf, df, I0, c, w, G, w_delta, max_wealth

    
    #returns
#    file_returns = '/Users/dduque/Dropbox/Northwestern/Research/Pension Funds DP/rentabilidad_real_mensual_fondos_deflactada_uf.xls'
#    returns_cl  = pd.read_excel(file_returns, skiprows = 2)
#    returns_cl =  returns_cl.dropna() 
#    
#    Funds = ['A','B','C','D','E']
#    monthly_returns = {f:np.array(returns_cl['Fondo Tipo %s' %f]/100) for f in Funds}
#    data = np.array([np.array(returns_cl['Fondo Tipo %s' %f]/100) for f in Funds]).transpose()
#    
##    if False: #Show correlogram
##        data_df = pd.DataFrame(data=data, columns=Funds)
##        for f in data_df.columns:
##            plot_pacf(data_df[f], lags=24)
##    invs_marginals, marg_names= fit_returns(data)
#    
#    #norta_data = fit_NORTA(data,len(data),len(data[0]), F_invs=invs_marginals)
#    #pickle.dump( norta_data.C, open('./norta_obj.pickle', 'wb'), pickle.HIGHEST_PROTOCOL)
#    norta_file = os.path.join(PF_path,'norta_obj.pickle' )
#    norta_c_matrix = pickle.load(open(norta_file, 'rb'))
#    F_invs = [build_empirical_inverse_cdf(np.sort(data[:,i])) for i in range(len(Funds))]
#    #F_invs = invs_marginals
#    norta_data = NORTA(F_invs ,norta_c_matrix )
#    NG = norta_data.gen(10000)
#    monthly_returns = {f:NG[:,i] for (i,f) in enumerate(Funds)}
#    r = gen_yearly_returns(Funds, monthly_returns,  n_years=1000)
#    #pickle.dump( r, open('./returns_train.p', 'wb'), pickle.HIGHEST_PROTOCOL)
#    
#    Funds = list(r.keys())
#    Funds.sort()
#    print([np.mean(r[a]) for a in Funds])
#    print([np.std(r[a]) for a in Funds])
#    print([(np.mean(r[a])-rf)/np.std(r[a]) for a in Funds])
#    #plt.scatter([np.std(r[a]) for a in Funds] , [np.mean(r[a]) for a in Funds] )
#    
#   
#    
#    np.random.seed(0)
#    replicas = 10000
#    simulated_returns = np.zeros((replicas,T,len(Funds)))
#    NG_sim =norta_data.gen(20000) # np.random.multivariate_normal(sr_mean_returs,data_cov,size=10000)#
#    monthly_returns_sim = {f:NG_sim[:,i] for (i,f) in enumerate(Funds)}
#    r_sim = gen_yearly_returns(Funds, monthly_returns_sim,  n_years=15000)
#    for k in range(replicas):
#        for t in range(T):
#            r_index = np.random.randint(0,len(r_sim['A']))
#            for (i,a) in enumerate(Funds):
#                simulated_returns[k,t,i] = r_sim[a][r_index]
#    #pickle.dump(simulated_returns, open('./returns_test.p', 'wb'), pickle.HIGHEST_PROTOCOL)
    
    return_train_file = os.path.join(PF_path, 'returns_train.p' )
    r = pickle.load(open(return_train_file, 'rb'))
    return_test_file = os.path.join(PF_path, 'returns_test.p' )
    simulated_returns = pickle.load(open(return_test_file, 'rb'))
    
    
    problem_params = T, R, rf, df, I0, c, w, G, w_delta, max_wealth
    dp_data = setup(T,r,w_delta,max_wealth)
    
    default_policy, default_sim = run_default(problem_params, dp_data, r,simulated_returns,plot=False) 
    Y =  np.array([sr[-1] for sr in default_sim])
    sols_DP = {'Default':(default_policy, default_sim)}
     #plot_policies_comparizon(('Default', Default_sim_results),('DP utility', DP_sim_results), G)
    
    
    setup_data = setup(T,r,w_delta,max_wealth)
    methods_dp = [ALG_UTILITY_POWER]#[ALG_UTILITY, ALG_CVAR_PENALTY, ALG_SSD, ALG_SSD_TAIL,ALG_SSD_MINMAX]  
    
    alg_params = (6, 0, 0)
    policy_name_params = '%s_%.2f_%.2f_%.2f'  %(methods_dp[0],alg_params[0], alg_params[1], alg_params[2])
      
    
    for m in methods_dp:
        dp_out = backward_induction_sd_mix_par(problem_params, dp_data, r, Y,default_policy, method=m, method_params = alg_params , method_cond=False, n_threads=3)
        V,U = dp_out
        w_map = setup_data[3]
        DP_sim_results = simulation(T,U,w_map,simulated_returns,I0,c, replicas , policy_name="%10s" %(m))
        sols_DP[m] = (dp_out, DP_sim_results)
    
    S, A, F, w_map, steps = setup_data 
    
    all_policies_out  = (S, A, F, T,r,w_delta,max_wealth,simulated_returns, sols_DP)
    out_path = os.path.join(PF_path,'%s.pickle' %(policy_name_params))
    pickle.dump(all_policies_out , open(out_path, 'wb'), pickle.HIGHEST_PROTOCOL)
  
    #Read solution 
    if False:
        PF_path = '/Users/dduque/MacWorkspace/PorfolioOpt/PensionFunds/'
        out_path = os.path.join(PF_path, 'utility_power_gamma4_out.pickle')
        read_out = pickle.load(open(out_path, 'rb')) 
        S, A, F, T,r,w_delta,max_wealth ,simulated_returns, sols_DP = read_out
        methods_dp = sols_DP.keys()
        w_map = w_index(w_delta, max_wealth,len(S))
        #scp dduque@crunch.osl.northwestern.edu:/home/dduque/dduque_projects/PorfolioOpt/PensionFunds/Plots/*.pdf ./PensionFunds/Plots/
        #scp dduque@crunch.osl.northwestern.edu:/home/dduque/dduque_projects/PorfolioOpt/PensionFunds/*.pickle ./PensionFunds/
        for m in methods_dp:
            dp_out, DP_sim_results = sols_DP[m]
            
            if m != 'Default':
                V,U = dp_out
                simulation(T,U,w_map,simulated_returns,I0,c, replicas , policy_name="%10s" %(m))
                plot_policy_and_sim2(T ,S, w_map, U, F, A, G, DP_sim_results, m)
                plot_policies_comparizon(('Default', sols_DP['Default'][1]),(m, DP_sim_results), G)
        
        
        #get_investment_profiles(DP_sim_results, U, w_map,T,A, Funds)
#    for t in range(T):
#        plt.plot(V[t,:])
#    for a in Funds:
#        policy_a = {(t,w_map(s)):[1,0,0,0,0] for t in range(T) for s in S}
#        sim_results = simulation(T,policy_a,w_map,simulated_returns,I0,c, replicas, policy_name="%10s" %(a))
#        #plot_policy_and_sim(T ,S, w_map,  policy_a, Funds, G, sim_results )
#        plot_policies_comparizon((a, sim_results),('', []), G)



if False:
    dp_name = ALG_UTILITY
    
        
    def_wealth = np.array([sols_DP['Default'][1][k][-1] for k in range(len(sols_DP['Default'][1]))])
    uti_wealth = np.array([sols_DP[dp_name][1][k][-1] for k in range(len(sols_DP[dp_name][1]))])
    
    check_SSD(uti_wealth, def_wealth)
    
    
    
    valid_reps = uti_wealth<G
    valid_reps1 = def_wealth<G
    total = valid_reps *  valid_reps1
    
    len(def_wealth[valid_reps1])
    
    valid_map = []
    for i in range(len(valid_reps)):
        if valid_reps[i]:
            valid_map.append(i)
            
    
    wealth_diff = uti_wealth[valid_reps] - def_wealth[valid_reps]
    #wealth_diff = uti_wealth - def_wealth
    best_def_replica = valid_map[np.argmin(wealth_diff)]
    basic_cols = ['red', 'blue', 'lime', 'magenta', 'yellow']
    fig, ax = plt.subplots()
    ax.grid(True)
    for (i,colf) in enumerate(basic_cols):
        ax.plot(simulated_returns[best_def_replica,:,i], color=colf)
    
    
    
    bins = 10
    wealth_diff1 = np.copy(wealth_diff)
    wealth_diff1.sort()
    heights,bins = np.histogram(wealth_diff1,bins=bins)
    n_pdf = 10000
    fig, ax = plt.subplots()
    ax.hist(wealth_diff,bins=50, weights=np.zeros_like(wealth_diff) + 1. / wealth_diff.size)
    ax.set_xlabel('Wealth difference (SSD-tail- Default)')
    ax.set_ylabel('Frequency')
#    hist_dist1 = scipy.stats.rv_histogram((heights,bins))
#    X = np.linspace(wealth_diff1[0], wealth_diff1[-1], n_pdf)
#    max_p1 = np.max(hist_dist1.pdf(X))
#    ax.fill_between(X,hist_dist1.pdf(X),np.zeros_like(X),  alpha=0.5)
    
    
    policy = sols_DP[dp_name][0][1]
    def_pol = sols_DP['Default'][0]
    basic_cols_rgs = np.array([col.to_rgb(c) for c in basic_cols])
    policy_colors = A.dot(basic_cols_rgs)
    actions_str = [str(pc) for pc in A]
    F_map = {a:i for (i,a) in enumerate(actions_str)}
    U_plot = np.zeros((2,T))
    def_pol_acts = np.array([def_pol[t,w_map(sols_DP[dp_name][1][best_def_replica][t])] for t in range(T)])
    dp_pol_acts = np.array([policy[t,w_map(sols_DP[dp_name][1][best_def_replica][t])] for t in range(T)])
    U_plot[0] = np.array([F_map[str(def_pol[t,w_map(sols_DP[dp_name][1][best_def_replica][t])])] for t in range(T)])
    U_plot[1] = np.array([F_map[str(policy[t,w_map(sols_DP[dp_name][1][best_def_replica][t])])] for t in range(T)])
    cmap = colors.ListedColormap(policy_colors)
    fig, ax = plt.subplots()
    steps_displayed = len(U_plot)
    im = ax.imshow(U_plot, cmap=cmap,  interpolation='none', aspect=T/steps_displayed ,origin='lower', vmin=0, vmax=len(A)-1)
     # We want to show all ticks...
    ax.set_xticks(np.arange(0,T+1,5))
    y_ticks = np.arange(0,len(quantiles),1)
    ax.set_yticks([0 , 1 ])
    ax.set_yticklabels(['Default', 'SSD-Tail'])
    ax.set_xlabel('Years')
    #ax.set_ylabel('Policy')
    plt.show()
    
    fig, ax = plt.subplots()
    ax.plot(sols_DP[dp_name][1][best_def_replica] , label='DP') 
    ax.plot(sols_DP['Default'][1][best_def_replica] , label='Default')
    ax.legend( loc=2)
    ax.set_xlabel('Years')
    ax.set_ylabel('Wealth (\$USD)')
    plt.tight_layout()
    
    #Conditinal distributions plot
    fig, ax = plt.subplots(figsize=(10, 6))
    mu = def_wealth.mean()
    variance = def_wealth.var()
    sigma = np.sqrt(variance)
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
    ax.plot(x,mlab.normpdf(x, mu, sigma), label='Default density')
    #density for dp policy
    uti_wealthT = np.array([sols_DP[dp_name][1][k][-2] for k in range(len(sols_DP[dp_name][1]))])
    sT_high = 1.5*G*(1+r['A']) + 21647.44*c
    sT_med = G*1.0*(1+r['A']) + 21647.44*c
    sT_low = G*0.7*(1+r['A']) + 21647.44*c
    ax.plot(x,mlab.normpdf(x, sT_low.mean(), sT_low.std()), label='SSD low')
    ax.plot(x,mlab.normpdf(x, sT_med.mean(), sT_med.std()), label='SSD medium')
    ax.plot(x,mlab.normpdf(x, sT_high.mean(), sT_high.std()), label='SSD high')
    ax.set_xlabel('Wealth at stage T+1' ,fontsize=12)
    ax.set_ylabel('Frequency' ,fontsize=12)
    ax.legend( loc='best')
    plt.tight_layout()
    
    
    
    
    #VaR and CVaR plots
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.style.use('default')
    mu = def_wealth.mean()
    variance = def_wealth.var()
    sigma = np.sqrt(variance)
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
    ax.plot(x,mlab.normpdf(x, mu, sigma), color='blue', label='pdf')
    rnd_nums = np.random.normal(mu,sigma,size=100000)
    alpha_ = 0.9
    var_a = np.percentile(rnd_nums,int(alpha_*100))
    cvar_a = cvar(rnd_nums, alpha_)
    ax.axvline(var_a , label='VaR', color='red')
    ax.axvline(cvar_a , label='CVaR', color='black')
    ax.set_xlabel('$X$' ,fontsize=12)
    ax.set_ylabel('Frequency' ,fontsize=12)
    ax.legend( loc='best')
    ax.annotate(r'$1-\alpha$', xy=(2.2E5, 0.8E-6), fontsize=20)

    plt.tight_layout()
    
    
    fig, ax = plt.subplots()
    mu = def_wealth.mean()
    variance = def_wealth.var()
    sigma = np.sqrt(variance)
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
    ax.plot(x,mlab.normcdf(x, mu, sigma), label='Default density')
    #density for dp policy
    uti_wealthT = np.array([sols_DP[dp_name][1][k][-2] for k in range(len(sols_DP[dp_name][1]))])
    sT_high = 1.5*G*(1+r['A']) + 21647.44*c
    sT_med = G*1.0*(1+r['A']) + 21647.44*c
    sT_low = G*0.7*(1+r['A']) + 21647.44*c
    ax.plot(x,mlab.normcdf(x, sT_low.mean(), sT_low.std()), label='SSD low')
    ax.plot(x,mlab.normcdf(x, sT_med.mean(), sT_med.std()), label='SSD medium')
    ax.plot(x,mlab.normcdf(x, sT_high.mean(), sT_high.std()), label='SSD high')
    ax.set_xlabel('Wealth at stage T+1')
    ax.set_ylabel('Frequency')
    ax.legend( loc='best')
    plt.tight_layout()