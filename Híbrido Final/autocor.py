# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 23:01:50 2019

@author: Guilherme
"""

import numpy as np
from pandas import Series

a = np.loadtxt('downjonesv2.txt')
b = Series(a)

from statsmodels.graphics.tsaplots import plot_acf

plot_acf(b,lags=300)