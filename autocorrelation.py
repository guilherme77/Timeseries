# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 13:18:14 2019

@author: Guilherme
"""

import numpy as np
from pandas import Series
a = np.loadtxt('credito.txt')
b = Series(a)

from statsmodels.graphics.tsaplots import plot_acf

plot_acf(b,lags=45)