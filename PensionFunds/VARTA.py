#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 14:15:06 2018

@author: dduque

Wrapper for VARTA method available at:
http://users.iems.northwestern.edu/~nelsonb/VARTA/varta.html
"""
import numpy as np
import pandas as pd
from scipy.stats import moment
from scipy.stats import kurtosis, skew
from scipy.stats import johnsonsu
import matplotlib.pyplot as plt 
import scipy.stats as st
from fitter import Fitter


def write_VARTA_input(data, lag, samples):
    '''
    Write a input file for VARTA following the format in 
    http://users.iems.northwestern.edu/~nelsonb/VARTA/Help.txt
    
    Args:
        data (ndarray): Data to be fit where the columns represent different
            random variables.
        lag (int): lag period to consider
    '''
    assert len(data)-1>lag, 'Lag need to be lower'
    d = len(data[0])
    n = len(data)
    rho_matrices = []
    for l in range(lag+1):
        if l == 0 :
            rho_l = 0.9*np.corrcoef(data,rowvar=0)+0.1*np.diag(np.diag(np.corrcoef(data,rowvar=0)))
            rho_matrices.append(rho_l)
        else:
            rho_l = np.zeros((d,d))
            for i in range(d):
                for j in range(d):
                    xi = data[:,i]
                    xj = data[:,j]
                    rho_l[i,j] = estimated_autocorrelation(xi,xj,l)
            rho_matrices.append(rho_l)
    
    f= open("../data/VARTA.dat","w")
    f.write('%i\n' %samples)
    f.write('%i\n' %d)
    f.write('%i\n' %lag)
    f.write('2\n')
    for k in range(d):
        f.write('%.8f\n' %data[:,k].mean())
        f.write('%.8f\n' %data[:,k].std())
        ske_k = skew(data[:,k])
        sqrt_beta1 = np.sign(ske_k)*np.sqrt(np.abs(ske_k))
        f.write('%.8f\n' %sqrt_beta1)
        beta2 = kurtosis(data[:,k])
        f.write('%.8f\n' %beta2)
    
    for l in range(lag+1):
        for i in range(d):
            for j in range(d):
                 f.write('%.8f\n' %(rho_matrices[l][i,j]))
    f.write('5\n')   
    f.close()
    
    return rho_matrices
                    
                    
        
    
    
def estimated_autocorrelation(x,y,l):
    """
    http://stackoverflow.com/q/14297012/190597
    http://en.wikipedia.org/wiki/Autocorrelation#Estimation
    """
    nx = len(x)
    ny = len(y)
    xl = x[:nx-l]
    yl = y[l:ny]
    sd_x = xl.std()
    sd_y = yl.std()
    covxy = np.sum((xl-xl.mean())*(yl-yl.mean()))/(nx-1-l)
    corxy = covxy/(sd_x*sd_y)
    return corxy


if __name__ == '__main__':
    file_returns = '/Users/dduque/Dropbox/Northwestern/Research/Pension Funds DP/rentabilidad_real_mensual_fondos_deflactada_uf.xls'
    returns_cl  = pd.read_excel(file_returns, skiprows = 2)
    returns_cl =  returns_cl.dropna() 
    
    Funds = ['A','B','C','D','E']
    monthly_returns = {f:np.array(returns_cl['Fondo Tipo %s' %f]/100) for f in Funds}
    
    data = np.array([np.array(returns_cl['Fondo Tipo %s' %f]) for f in Funds]).transpose()
    
    matrices = write_VARTA_input(data,lag=0,samples=10000)
    print(matrices)