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
from scipy.spatial import ConvexHull
rnd  = np.random.RandomState(0)
Z = stats.norm(0,1)
GAUSS_SAMPLING = True
def reset_strem():
    rnd.seed(0)



def find_rho_z(i,j,F_invs, CovX, EX):
    cor_dem = np.sqrt(CovX[i,i]*CovX[j,j])
    rho_z = CovX[i,j]/cor_dem
    rho_u = 1 if CovX[i,j]>0 else 0
    rho_d = -1 if CovX[i,j]<0 else 0
    F_i_inv = F_invs[i]
    F_j_inv = F_invs[j]
    EXi , EXj = EX[i], EX[j]
    while np.abs(rho_u - rho_d)>1E-4:
        print('testing ' , rho_z )
        covZ = np.array([[1,rho_z],[rho_z,1]])
        f = conv_exp(covZ,F_i_inv, F_j_inv, gaussian_sampling=GAUSS_SAMPLING)
        #c_ij = montecarlo_integration(f, n=2000000000) - EXi*EXj #dblquad(f, -6, 16, lambda z2: -16, lambda z2: 6)
        EXiXj = montecarlo_integration(f, n=10000000, c = covZ, m=np.zeros(2), gaussian_sampling=GAUSS_SAMPLING)
        CXiXj = EXiXj - EXi*EXj
        if np.abs(CXiXj - CovX[i,j])/cor_dem < 1E-4:
            print(rho_z, '-->' , CXiXj, '--->' , CovX[i,j])
            return rho_z
        else:
            if CXiXj > CovX[i,j]:
                rho_u = rho_z
                rho_z = 0.5*(rho_z+rho_d)
            else: # rhoC_ij <= rho_ij
                rho_d = rho_z
                rho_z = 0.5*(rho_z+rho_u)
    print(rho_z, '-->' , CXiXj, '--->' , CovX[i,j])
    return rho_z


def montecarlo_integration(f, m=None,  c = None, n = 1000000,gaussian_sampling =False):
    if gaussian_sampling:
        assert type(m)!=type(None), 'Mean and Cov are required for gaussian sampling'
        z_trial = np.random.multivariate_normal(m,c,n)
        integral = np.sum(f(z_trial[:,0], z_trial[:,1]))
        return integral/n
    else:
        return montecarlo_integration_uniform(f,n)
    

def montecarlo_integration_uniform(f, n = 1000):
    cube_size = 5
    z1_trials = np.random.uniform(-cube_size,cube_size,n)#np.random.normal(0,1,n)
    z2_trials = np.random.uniform(-cube_size,cube_size,n)#np.random.normal(0,1,n)
    V = 2*cube_size*2*cube_size#(np.max(z1_trials) - np.min(z1_trials)) * (np.max(z2_trials) - np.min(z2_trials)) 
    integral = np.sum(f(z1_trials, z2_trials))
    return V*integral/n

def PolyArea(x,y):
    #https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))   
    
def conv_exp(covZ, F_i_inv, F_j_inv, gaussian_sampling=True):
    if gaussian_sampling:
        def f(z1,z2):
            return F_i_inv(Z.cdf(z1))*F_j_inv(Z.cdf(z2)) # remove bi_x pdf since montecarlo is sampling according to  bi_z
        return f
    else:
        print('hola')
        bi_z = stats.multivariate_normal(cov=covZ)
        def f(z1,z2):
            z1z2 = np.vstack((z1,z2)).transpose()
            return F_i_inv(Z.cdf(z1))*F_j_inv(Z.cdf(z2))*bi_z.pdf(z1z2)
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
    '''
    Computes covaianze matrix for NORTA algorith.
    
    Args:
        data (ndarray): a n x d array with the data
        n (int): number of observations for each random variable
        d (int): dimanention of the random vector.
        F_invs (list of func): optional parameter to specify the marginal 
            distributions. Default is None, which constructs the marginals from
            the data.
    Return:
        NORTA_GEN (NORTA): an object that stores the necessary information to 
            generate NORTA random vectors. 
    '''
    assert len(data) == n, 'Data needs to be a d x n matrix'
    assert len(data[0]) == d, 'Data needs to bo a d x n matrix'
    C = None # matrix for NORTA
    lambda_param = 0.01
    CovX = np.cov(data,rowvar=False)
    VarX = np.diag(np.diag(CovX))
    procedure_done = False
    while procedure_done==False:
        D = np.eye(d)
        EX = np.mean(data, axis = 0)
        if type(F_invs) != list:
            F_invs = [build_empirical_inverse_cdf(np.sort(data[:,i])) for i in range(d)]
        print('Finding %i correlation terms' %(int(d*(d-1)/2)))
        for i in range(d):
            for j in range(i+1,d):
                D[i,j]= find_rho_z(i,j,F_invs,CovX,EX)
                D[j,i]= D[i,j]
        try:
            C = cholesky(D, lower=True)
            procedure_done = True
        except:
            CovX = (1-lambda_param)*CovX + lambda_param*VarX
            print('Cholesky factorization failed, starting over')
    
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
    reset_strem()
    n_sample = 100
    d_sample = 2
    cov_sample = np.eye(d_sample) + np.random.rand(d_sample,d_sample)
    sim_cov = cov_sample.transpose().dot(cov_sample)
    data = np.random.exponential(size=(n_sample,d_sample)) + np.random.multivariate_normal(np.zeros(d_sample),sim_cov,size=n_sample)
    n = len(data)
    d = len(data[0])
    norta_data = fit_NORTA(data,n,d)
    NG = norta_data.gen(1000000)
    print(NG.mean(axis=0), data.mean(axis=0))
    print(np.corrcoef(NG,rowvar=0))
    print(np.corrcoef(data,rowvar=0))
    print(np.cov(NG,rowvar=False))
    print(np.cov(data,rowvar=False))
    

                
                
                
      