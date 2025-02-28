# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 17:52:41 2019

@author: Guilherme
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor

def gridRF(X_train,Y_train,X_val,Y_val):
    
    """
    Observação: Alto custo de CPU
    """
    
    n_estimators = [200]
    max_features = ['auto','sqrt','log2',None]
    min_samples_leaf = np.linspace(0.001, 0.5, 2)
    min_samples_split = np.linspace(0.001, 1.0, 2)
    bootstrap = [True,False]
    
    best_erro = 9999999999999999999999999999999999999999**99999
    
    for nest in n_estimators:
        for mf in max_features:
            for msl in min_samples_leaf:
                for msp in min_samples_split:
                    for bts in bootstrap:
                    
                        rf = RandomForestRegressor(n_estimators = nest, max_features = mf, min_samples_leaf = msl, min_samples_split = msp, bootstrap = bts)
                        rf.fit(X_train, Y_train)
                        
                        predict = rf.predict(X_val)
                        erro = metrics.mean_squared_error(Y_val, predict)
                        
                        if erro < best_erro:
                            
                            best_erro = erro
                            best_model = rf
                            best_predicts = predict
                            best_param = (nest,mf,msl,msp)
                            
                            print best_param
                            print best_erro
                            
            
    return (best_model, best_predicts, best_erro, best_param)

def processdata(datafile,dimension, valid = True):

    data = np.loadtxt(datafile)
    serie = pd.Series(data)
    laggedata = pd.concat([serie.shift(i) for i in range(dimension+1)],axis=1 )

    if valid == False: 
        
        #Treinamento
        trainset = laggedata.iloc[dimension:int(np.floor(0.8*len( laggedata))),1:dimension+1]
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
    if opmodel == 'RF':
        (best_model, best_predicts, best_erro, best_param) = gridRF(trainset,traintarget,valset,valtarget)
        return best_model
    
  

def predict(testset,testtarget,model):
  
    predicts = model.predict(testset)
    erro = metrics.mean_squared_error(testtarget,predicts)

    return (predicts,erro)        

dimension = 13
datafile = 'downjonesv2.txt'

(data_set, data_target, val_set, val_target, pred_set, pred_target) = processdata(datafile , dimension)

#RandomForest

rf_reg = createmodel(data_set, data_target, val_set, val_target, 'RF')
(pred_rf, erro_rf) = predict(pred_set, pred_target, rf_reg)


print 'Modelos'
print 'RF  Erro: %.4f' %erro_rf


#Plotagem
x = range(len(pred_target))

plt.plot(x, pred_target, 'r--', label = 'Real')
plt.plot(x, pred_rf, label = 'RF predicted')
plt.legend()
plt.figure
