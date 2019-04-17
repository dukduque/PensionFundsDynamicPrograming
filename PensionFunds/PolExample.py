#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 16:34:33 2019

@author: dduque
"""


from PensionFunds.NORTA import fit_NORTA, NORTA, build_empirical_inverse_cdf
import numpy as np 

data = np.random.gamma(3, size=(10_000,2))



NORTA_OBJECT = fit_NORTA(data, 10_000, 2 )