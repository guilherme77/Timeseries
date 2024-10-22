# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 18:38:12 2019

@author: Guilherme
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn import svm

C_r = [10**(-1), 10**(0), 10**(1), 10**(2)]
Epsilon_r = [10**(-4), 10**(-3), 10**(-2)]
Gamma_r = [2**(-5), 2**(-4), 2**(-3), 2**(-2), 2**(-1), 2**(0), 2**(1), 2**(2)]
    
def gridSVR(X_train,Y_train,X_val,Y_val):
   
    bestModel = 1
    bestError = 99999999999999999999999999999999999999.0
    bestPredicts = 1
    bestParam = (0,0,0)
    
    for c in C_r:
        for e in Epsilon_r:
            for g in Gamma_r:
                model = svm.SVR(kernel = 'linear', C=10**c,epsilon=10**e,gamma=10**g)
                model.fit(X_train,Y_train)
                predicts = model.predict(X_val)
                erro = metrics.mean_squared_error(Y_val,predicts)
                
                if (erro < bestError):
                    bestError = erro
                    bestModel = model
                    bestPredicts = predicts
                    bestParam = (c,e,g)
                    print bestError
                    print bestParam
                    
    return (bestModel,bestPredicts,bestError,bestParam)

def processdata(datafile,dimension, valid = True):

    data = np.loadtxt(datafile)
    serie = pd.Series(data)
    laggedata = pd.concat([serie.shift(i) for i in range(dimension+1)],axis=1 )

    if valid == False: 
        
        #Treinamento 80%
        trainset = laggedata.iloc[dimension:int(np.floor(0.8*len( laggedata))),1:dimension+1]
        traintarget = laggedata.iloc[dimension:int(np.floor(0.8*len( laggedata))),0]
        
        #Teste 20%
        testset =  laggedata.iloc[int(np.floor(0.8*len( laggedata))):len( laggedata),1:dimension+1]
        testtarget =  laggedata.iloc[int(np.floor(0.8*len( laggedata))):len( laggedata),0]
    
        return (trainset,traintarget,testset,testtarget)
    
    if valid == True:
        
        trainindex=int(np.floor(0.7*len(data)))
        valindex=int(np.floor(0.8*len(data)))
         
        #Treinamento 60%
        trainset = laggedata.iloc[dimension:trainindex,1:dimension+1]
        traintarget = laggedata.iloc[dimension:trainindex,0]
        
        #Validação 20%
        valset = laggedata.iloc[trainindex:valindex,1:dimension+1]
        valtarget = laggedata.iloc[trainindex:valindex,0]
        
        #Teste 20%
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

print 'SVR  Erro: %.4f' %erro_svr

#Plotagem
x = range(len(pred_target))

#3 Modelos
plt.plot(x, pred_target, 'r--', label = 'Real')
plt.plot(x, pred_svr, label = 'SVR predicted')
plt.legend()
plt.figure
