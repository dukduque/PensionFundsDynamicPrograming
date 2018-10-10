#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 12:58:58 2018

@author: dduque
"""
import numpy as np
import pandas as pd
from scipy.linalg import cholesky
import scipy.stats as stats
from scipy.integrate import dblquad
rnd  = np.random.RandomState(0)
Z = stats.norm(0,1)

def reset_strem():
    rnd.seed(0)



def find_rho_z(i,j,F_invs, CovX, EX):
    rho_ij = CovX[i,j]/np.sqrt(CovX[i,i]*CovX[j,j])
    rho_z = rho_ij
    rho_u = 1 if rho_ij>0 else 0
    rho_d = -1 if rho_ij<0 else 0
    F_i_inv = F_invs[i]
    F_j_inv = F_invs[j]
    EXi , EXj = EX[i], EX[j]
    while np.abs(rho_u - rho_d)>1E-4:
        print('testing ' , rho_z )
        covZ = np.array([[1,rho_z],[rho_z,1]])
        f = conv_exp(covZ,F_i_inv, F_j_inv)
        result = montecarlo_integration(f, n=10000000)#dblquad(f, -6, 16, lambda z2: -16, lambda z2: 6)
        rhoC_ij = (result - EXi*EXj)/np.sqrt(CovX[i,i]*CovX[j,j])
        if np.abs(rhoC_ij - rho_ij) < 1E-4:
            print(rho_z, '-->' , rhoC_ij, '--->' , rho_ij)
            return rho_z
        else:
            if rhoC_ij > rho_ij:
                rho_u = rho_z
                rho_z = 0.5*(rho_z+rho_d)
            else: # rhoC_ij <= rho_ij
                rho_d = rho_z
                rho_z = 0.5*(rho_z+rho_u)
    print(rho_z, '-->' , rhoC_ij, '--->' , rho_ij)
    return rho_z

def montecarlo_integration(f, n = 1000):
    
    cube_size = 5
    z1_trials = np.random.uniform(-cube_size,cube_size,n)#np.random.normal(0,1,n)
    z2_trials = np.random.uniform(-cube_size,cube_size,n)#np.random.normal(0,1,n)
    V = 2*cube_size*2*cube_size#(np.max(z1_trials) - np.min(z1_trials)) * (np.max(z2_trials) - np.min(z2_trials)) 
    integral = np.sum(f(z1_trials, z2_trials))
    return V*integral/n
    
    
def conv_exp(covZ, F_i_inv, F_j_inv):
    bi_z = stats.multivariate_normal(cov=covZ)
    def f(z1,z2):
        z1z2 = np.vstack((z1,z2)).transpose()
        return F_i_inv(Z.cdf(z1))*F_j_inv(Z.cdf(z2))*bi_z.pdf(z1z2)
        #return bi_z.pdf(z1z2)
    return f

def build_empirical_inverse_cdf(X):
    '''
    Builds an inverse CDF function given a sorted vector of values defining a 
    marginal distribution.
    
    Args:
        X:Sorted vector of observations
    '''
    n = len(X)
    def f(prob):
        '''
        Args:
            prob (ndarray): vector with probablities to compute the inverse
        '''
        #assert 0<=prob<=1, 'Argument of inverse function is a probability >=0 and <= 1.'
        return X[np.minimum((n*np.array(prob)).astype(int),n-1)]
    return f

def fit_NORTA(data,n,d, F_invs=None):
    assert len(data) == n, 'Data needs to be a d x n matrix'
    assert len(data[0]) == d, 'Data needs to bo a d x n matrix'
    
    CovX = np.cov(data,rowvar=False)
    Corr = np.corrcoef(data,rowvar=0)
    D = np.eye(d)
    EX = np.mean(data, axis = 0)
    if type(F_invs) != list:
        F_invs = [build_empirical_inverse_cdf(np.sort(data[:,i])) for i in range(d)]
    for i in range(d):
        for j in range(i+1,d):
            D[i,j]= find_rho_z(i,j,F_invs,CovX,EX)
            D[j,i]= D[i,j]
    
    C = cholesky(D)
    
    NORTA_GEN = NORTA(F_invs, C)
    return NORTA_GEN
    
    
class NORTA():
    '''
    Class to create a Normal-to-Anything model
    Attributes:
        F (list of func): Marginal inverse CDFs
        C (ndarry): numpy array with the Cholesky factorization
    '''    
    def __init__(self, Finv, C):
        self.F_inv = Finv
        self.C = C
        
    
    
    def gen(self, n = 1):
        d = len(self.F_inv)
        w = rnd.normal(size=(d,n))
        z = self.C.dot(w)
        
        X = np.array([self.F_inv[i](Z.cdf(z[i])) for i in range(d)]).transpose()
        return X
        

if __name__ == "__main__":
    assets = pd.read_excel('./data/bancolombia/Consolidado.xlsx', sheet_name='asset_daily_returns')
    assets = assets.dropna() 
    data = np.array(assets.iloc[:,1:])
    n = len(data)
    d = len(data[0])
    norta = fit_NORTA(data,n,d)
    NG = NORTA_GEN.gen(10000)
    print(NG.mean(axis=0), EX)
    print(np.corrcoef(NG,rowvar=0))
    print( Corr)
    

                
                
                
      