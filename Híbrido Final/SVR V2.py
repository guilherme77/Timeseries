# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 21:19:17 2019

@author: Guilherme
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn import svm

Cr = [10**(-1), 10**(0), 10**(1), 10**(2)]
epsilonr = [10**(-4), 10**(-3), 10**(-2)]
gammar = [2**(-5), 2**(-4), 2**(-3), 2**(-2), 2**(-1), 2**(0), 2**(1), 2**(2)]
    
def gridSVR(X_trein,Y_trein,X_val,Y_val):
   bestParam=(0,0,0)
   bestModel= 1
   bestError=99999999999999999999999999999999999999999999
   bestPredicts = 1
   
   for c in Cr:
       for eps in epsilonr:
           for gam in gammar:
               svr = svm.SVR(kernel = 'linear',gamma=10**gam, C=10**c, epsilon=10**eps)
               svr.fit(X_trein,Y_trein)
               predicts=svr.predict(X_val)
               mse = metrics.mean_squared_error(predicts,Y_val)
               
               if(mse<bestError):
                   bestParam = (c,eps,gam)
                   bestModel = svr
                   bestError = mse
                   bestPredicts = predicts
                   print mse
                   print bestError
                   print bestParam
                   
   return (bestModel,bestPredicts,bestError,bestParam)
               

def processdata(datafile,dimension, valid = True):

    data = np.loadtxt(datafile)
    serie = pd.Series(data)
    laggedata = pd.concat([serie.shift(i) for i in range(dimension+1)],axis=1 )

    if valid == False: 
        
        #Treinamento
        trainset = laggedata.iloc[dimension:int(np.floor(0.7*len( laggedata))),1:dimension+1]
        traintarget = laggedata.iloc[dimension:int(np.floor(0.8*len( laggedata))),0]
        
        #Teste
        testset =  laggedata.iloc[int(np.floor(0.7*len( laggedata))):len( laggedata),1:dimension+1]
        testtarget =  laggedata.iloc[int(np.floor(0.8*len( laggedata))):len( laggedata),0]
    
        return (trainset,traintarget,testset,testtarget)
    
    if valid == True:
        
        trainindex=int(np.floor(0.7*len(data)))
        valindex=int(np.floor(0.8*len(data)))
         
        #Treinamento
        trainset = laggedata.iloc[dimension:trainindex,1:dimension+1]
        traintarget = laggedata.iloc[dimension:trainindex,0]
        
        #Validação
        valset = laggedata.iloc[trainindex:valindex,1:dimension+1]
        valtarget = laggedata.iloc[trainindex:valindex,0]
        
        #Teste
        testset = laggedata.iloc[valindex:len(data),1:dimension+1]
        testtarget =  laggedata.iloc[valindex:len(data),0]
        
        return (trainset,traintarget,valset,valtarget,testset,testtarget)


def createmodel(trainset,traintarget,valset,valtarget, opmodel):
    
    if opmodel == 'SVR':
        (best_model, best_predicts, best_erro, best_param) = gridSVR(trainset,traintarget,valset,valtarget)
        return best_model
    
def predict(testset,testtarget,model):
  
    predicts = model.predict(testset)
    erro = metrics.mean_squared_error(testtarget,predicts)

    return (predicts,erro)

dimension = 1
datafile = 'credito.txt'

(data_set, data_target, val_set, val_target, pred_set, pred_target) = processdata(datafile , dimension)


#SVR

svr_reg = createmodel(data_set, data_target, val_set, val_target, 'SVR')
(pred_svr, erro_svr) = predict(pred_set, pred_target, svr_reg)

print 'SVR  Erro:' %erro_svr

#Plotagem
x = range(len(pred_target))

#3 Modelos
plt.plot(x, pred_target, 'r--', label = 'Real')
plt.plot(x, pred_svr, label = 'SVR predicted')
plt.legend()
plt.figure
