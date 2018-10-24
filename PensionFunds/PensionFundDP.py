#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 12:16:05 2018

@author: dduque
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import scipy.stats
import pandas as pd
import pickle
from matplotlib import colors
from PensionFunds.NORTA import fit_NORTA, NORTA, build_empirical_inverse_cdf

from statsmodels.distributions.empirical_distribution import ECDF

def utility_function(x, G):
    '''
    params:
        x (float): wealth
        G (float): target wealth
    '''
#    s1 = 10
#    s2 = 1
#    if x<=G:
#        return s1*(x-G) 
#    else:
#        return s2*(x - G)
    
    
#    u_gamma = 0.2
#    return (x/G)**u_gamma
    side_g = x<G
    exp_param = -0.1
    numerator = 101
    denominator = 100
    left = numerator/(denominator+np.exp((x-G)/(exp_param*G)))
    right = np.exp((x-G)/G)
    return left#side_g*left + (1-side_g)*right
   
    

def w_index(w_delta, max_wealth, steps):
    def f(x):
        p = np.round(x,0)/w_delta
        p = np.minimum(p, steps-1)
        p = np.maximum(p, 0)
        p = p.astype(int)
        return p
    return f 
    

def backward_induction(T, G , df, rf, r, I, c , steps, cvar = False):
    '''
    Runs backward induction to find the optimal policy of selecting 
    a fund at each time period give a particular amounth of wealth.
    '''
    assert df<1
    max_wealth = np.round(3*G,-3) - (np.round(3*G,-3) % (steps-1))
    w_delta = int(max_wealth/(steps-1)) 
    V = np.zeros((T+1,steps))
    U = {}
    
    
    
    A = r.keys()
    w_map = w_index(w_delta, max_wealth,steps)
    S = np.array([i*w_delta for i in range(steps) if i*w_delta<=max_wealth])
    
    for (i,s) in enumerate(S):
        assert w_map(s)==i, "NUMS %i %i %i" %(i,s,w_map(s))
    
    # Value in the last stage
    V[T,:] = utility_function(S,G) 

    for t in np.arange(T-1, -1, -1):
        I_t = I*(1+rf)**t
        print('solving ', t, ' ' , I_t)
        for s in S:
            #print('s=',w_map(s), ' ', s)
            s_index = w_map(s)
            arg_max = None
            V_s = -np.inf
            for a in A:
                v_a = None
                if t == T-1 and cvar==True: #CVAR
                    s_a_i = w_map((s)*(1+r[a])+c*I_t)
                    r_money = V[t+1,s_a_i]
                    shortfall = np.maximum(1-r_money,0)
                    if len(shortfall)>0:
                        v_a = -shortfall.mean() #Shortfall bellow G
                    else:
                        v_a = 0
                else:
                    s_a_i = w_map((s)*(1+r[a])+c*I_t)
                    v_a = df*(1/len(r[a]))*np.sum(V[t+1,s_a_i]) #Expectation
                if v_a>=V_s: #Choose less risk of alternative optima
                    V_s = v_a
                    arg_max = a
            V[t,s_index] = V_s
            U[t,s_index] = arg_max    
    return V,U,S,w_map

def simulation(T,U,w_map,r,I0,c, replications, fix_policy=None, policy_name =""):
    np.random.seed(0)

    I = [I0*(1+rf)**t for t in range(T)]
    wealth_sims = []
    for k in range(replications):
        wealth_k = [0]
        for t in range(T):
            policy = U[t,w_map(wealth_k[t])] if fix_policy==None else fix_policy
            wealth_k.append((1+r[k,t,policy])*wealth_k[t] + c*I[t])
            
        wealth_sims.append(wealth_k)
    if fix_policy == None:
        fix_policy = 'DP'
    wealths = np.array([sr[-1] for sr in wealth_sims])
    unmet_fraction = wealths[wealths <= G].size / wealths.size
    print('Policy: ' , policy_name, ' Unmet_frac:' , unmet_fraction , \
          ' Short:' , G-np.mean(wealths[wealths <= G]), ' EV: ' ,np.mean(wealths),\
          ' SD: ' ,np.std(wealths))
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
#    heights = heights/sum(heights)
#    ax.bar(bins[:-1],heights,width=(max(bins) - min(bins))/len(bins), color="blue", alpha=0.6 , label=p1_results[0])
    
    n_pdf = 10000
    fig, ax = plt.subplots()
    hist_dist = scipy.stats.rv_histogram((heights,bins))
    X = np.linspace(wealths[0], wealths[-1], n_pdf)
    max_p1 = np.max(hist_dist.pdf(X))
    ax.fill_between(X,hist_dist.pdf(X),np.zeros_like(X),  alpha=0.5)
    
    axR = ax.twinx()
    axR.plot(X, hist_dist.cdf(X), label=p1_results[0])
    
    
    wealths = [sr[-1] for sr in p2_results[1]]
    wealths.sort()
    bins = 300
    heights,bins = np.histogram(wealths,bins=bins)
#    heights = heights/sum(heights)
#    ax.bar(bins[:-1],heights,width=(max(bins) - min(bins))/len(bins), color="red", alpha=0.6, label=p2_results[0])
#    ax.legend(loc='best', shadow=True, fontsize='small')
#    ax.set_xlabel('Wealth at T')
#    ax.set_ylabel('Frequency')
#    ax.axvline(G , label='G')
#    axR.hist(wealths[:], bins=bins, normed=True, cumulative=True, label='CDF %s' %(p2_results[0]), histtype='step', alpha=0.8, color='red')
#    axR.set_yticks(np.linspace(0, 1, 5))
#    axR.set_ylim(0,1)
    hist_dist = scipy.stats.rv_histogram((heights,bins))
    X = np.linspace(wealths[0], wealths[-1], n_pdf)
    max_p2 = np.max(hist_dist.pdf(X))
    ax.fill_between(X,hist_dist.pdf(X),np.zeros_like(X), alpha=0.5, color='red')
    axR.plot(X, hist_dist.cdf(X), label=p2_results[0], color='red')
    
    
    ax.set_yticks(np.linspace(0, np.maximum(max_p1,max_p2), 11))
    ax.set_ylim(0,np.maximum(max_p1,max_p2))
    ax.set_ylabel('Frecuency')
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
    
def plot_policy_and_sim(T, S, w_map, policy, Funds, G, sim_results):
    F = {a:i for (i,a) in enumerate(Funds)}
    U_plot = np.array([[F[policy[t,w_map(s)]] for t in range(T)] for s in S if s<=G*1.5])
    U_plot[w_map(G)-6:w_map(G)+6,:] = -1 #Draw G
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
    
#    show_steps = int(steps/2)
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
    
    df = 0.98
    steps = 5001
    #returns
    
    file_returns = '/Users/dduque/Dropbox/Northwestern/Research/Pension Funds DP/rentabilidad_real_mensual_fondos_deflactada_uf.xls'
    returns_cl  = pd.read_excel(file_returns, skiprows = 2)
    returns_cl =  returns_cl.dropna() 
    
    Funds = ['A','B','C','D','E']
    monthly_returns = {f:np.array(returns_cl['Fondo Tipo %s' %f]/100) for f in Funds}
    
#    data = np.array([np.array(returns_cl['Fondo Tipo %s' %f]) for f in Funds]).transpose()
#    #norta_data = fit_NORTA(data,len(data),len(data[0]))
#    #pickle.dump( norta_data.C, open('./norta_obj.pickle', 'wb'), pickle.HIGHEST_PROTOCOL)
#    norta_c_matrix = pickle.load(open('./norta_obj.pickle', 'rb'))
#    F_invs = [build_empirical_inverse_cdf(np.sort(data[:,i])) for i in range(len(Funds))]
#    norta_data = NORTA(F_invs ,norta_c_matrix )
#    NG = norta_data.gen(10000)
#    monthly_returns = {f:NG[:,i]/100 for (i,f) in enumerate(Funds)}
    r = gen_yearly_returns(Funds, monthly_returns,  n_years=500)
    
    Funds = list(r.keys())
    Funds.sort()
    print([np.mean(r[a]) for a in Funds])
    print([np.std(r[a]) for a in Funds])
    
    
    
    np.random.seed(0)
    replicas = 10000
    simulated_returns = {}
    for k in range(replicas):
        for t in range(T):
            r_index = np.random.randint(0,len(r['A']))
            for a in Funds:
                simulated_returns[k,t,a] = r[a][r_index]
    
    
    V,U,S,w_map = backward_induction(T,G,df,rf,r,I0,c, steps , cvar=False)
    DP_sim_results = simulation(T,U,w_map,simulated_returns,I0,c, replicas , policy_name="%10s" %("DP utility"))
    #plot_simulation(DP_sim_results, U, style=0)
    plot_policy_and_sim(T ,S, w_map, U, Funds, G, DP_sim_results )
    
    def_pol = {(t,w_map(s)):('B' if t<=14 else ('C' if 14<t<=34 else 'D')) for t in range(T) for s in S}
    Default_sim_results = simulation(T,def_pol,w_map,simulated_returns,I0,c, replicas, policy_name="%10s" %("Default"))
    #plot_simulation(Default_sim_results, def_pol, style=1)
    plot_policy_and_sim(T ,S, w_map,  def_pol, Funds, G, Default_sim_results )
    
    plot_policies_comparizon(('Default', Default_sim_results),('DP utility', DP_sim_results), G)
    
    V,U,S,w_map = backward_induction(T,G,df,rf,r,I0,c, steps , cvar=True)
    DP_sim_results = simulation(T,U,w_map,simulated_returns,I0,c, replicas , policy_name="%10s" %("DP shortfall"))
    #plot_simulation(DP_sim_results, U, style=1)
    plot_policy_and_sim(T ,S, w_map, U, Funds, G, DP_sim_results )
    
    plot_policies_comparizon(('Default', Default_sim_results),('DP shortfall', DP_sim_results), G)
    
    
    for a in Funds:
        policy_a = {(t,w_map(s)):a for t in range(T) for s in S}
        sim_results = simulation(T,policy_a,w_map,simulated_returns,I0,c, replicas, policy_name="%10s" %(a))
        #plot_policy_and_sim(T ,S, w_map,  policy_a, Funds, G, sim_results )
        #plot_policies_comparizon(('', []),  (a, sim_results), G)
#    
#    fig, ax = plt.subplots()
#    heights,bins = np.histogram(rA,bins=20)
#    heights = heights/sum(heights)
#    ax.bar(bins[:-1],heights,width=(max(bins) - min(bins))/len(bins), color="red", alpha=0.6 , label='rA')



#fig, ax = plt.subplots()
#heights,bins = np.histogram(r['A'],bins=30)
#heights = heights/sum(heights)
#ax.bar(bins[:-1],heights,width=(max(bins) - min(bins))/len(bins), color="red", alpha=0.6)
