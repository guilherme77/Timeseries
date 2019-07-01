# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 22:14:06 2019

@author: Guilherme
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.neighbors import KNeighborsRegressor 

def gridKN(X_train,Y_train,X_val,Y_val):

    n_neighbors = range(1,10,10)
    weights = ['uniform','distance']
        
    best_erro = 9999999999999999999999999999999999

    for ngh in n_neighbors:
        for w in weights:
            kn = KNeighborsRegressor(n_neighbors = ngh, weights = w, algorithm = 'auto', p = 2)
            kn.fit(X_train,Y_train)

            predict = kn.predict(X_val)
            erro = metrics.mean_squared_error(Y_val, predict)

            if erro < best_erro:

                best_erro = erro
                best_model = kn
                best_param = (ngh, w)
                best_predicts = predict

    return (best_model, best_predicts, best_erro, best_param)

def processdata(datafile,dimension, valid = True):

    data = np.loadtxt(datafile)
    serie = pd.Series(data)
    laggedata = pd.concat([serie.shift(i) for i in range(dimension+1)],axis=1 )

    if valid == False: 
        
        #Treinamento
        trainset = laggedata.iloc[dimension:int(np.floor(0.7*len( laggedata))),1:dimension+1]
        traintarget = laggedata.iloc[dimension:int(np.floor(0.8*len( laggedata))),0]
        
        #Teste
        testset =  laggedata.iloc[int(np.floor(0.8*len( laggedata))):len( laggedata),1:dimension+1]
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
    if opmodel == 'KN': 
        (best_model, best_predicts, best_erro, best_param) = gridKN(trainset,traintarget,valset,valtarget)
        return best_model
  

def predict(testset,testtarget,model):
  
    predicts = model.predict(testset)
    erro = metrics.mean_squared_error(testtarget,predicts)

    return (predicts,erro)        

dimension = 1
datafile = 'credito.txt'

(data_set, data_target, val_set, val_target, pred_set, pred_target) = processdata(datafile , dimension)

#Kneighbors

kn_reg = createmodel(data_set, data_target, val_set, val_target, 'KN')
(pred_kn, erro_kn) = predict(pred_set, pred_target, kn_reg)



print 'Modelos'
print 'KN  Erro: %.4f' %erro_kn


#Plotagem
x = range(len(pred_target))

plt.plot(x, pred_target, 'r--', label = 'Real')
plt.plot(x, pred_kn, label = 'kn predicted')
plt.legend()
plt.figure