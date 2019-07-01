# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 17:28:32 2019

@author: Guilherme
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.neural_network import MLPRegressor

def gridMLP(X_train,Y_train,X_val,Y_val):      
                     
    """
    Melhores parametros encontrados do laço for abaixo:
        
    ('relu', 'lbfgs', 0.00055)
    """
    
    activation = ['relu','tanh']
    solver = ['lbfgs']
    alpha = np.linspace(0.0001, 0.001, 5)
    
    best_erro = 999999999999999999999999999999999999999999999999999999999999999999999**999

    
    for act in activation:
        for slv in solver:
            for aph in alpha:
                mlp = MLPRegressor(activation = act, solver = slv, alpha = aph)
                mlp.fit(X_train, Y_train)
                
                predict = mlp.predict(X_val)
                erro = metrics.mean_squared_error(Y_val, predict)
                
                if erro < best_erro:
                    
                    best_erro = erro
                    best_model = mlp
                    best_param = (act, slv, aph)
                    best_predicts = predict
                    
               #   print best_param
                #  print best_erro
                    
    
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
         
        #Treinamento 80
        trainset = laggedata.iloc[dimension:trainindex,1:dimension+1]
        traintarget = laggedata.iloc[dimension:trainindex,0]
        
        #Validação 20
        valset = laggedata.iloc[trainindex:valindex,1:dimension+1]
        valtarget = laggedata.iloc[trainindex:valindex,0]
        
        #Teste 20
        testset = laggedata.iloc[valindex:len(data),1:dimension+1]
        testtarget =  laggedata.iloc[valindex:len(data),0]
        
        return (trainset,traintarget,valset,valtarget,testset,testtarget)


def createmodel(trainset,traintarget,valset,valtarget, opmodel):
    if opmodel == 'MLP':
        (best_model, best_predicts, best_erro, best_param) = gridMLP(trainset,traintarget,valset,valtarget)
        return best_model
  


def predict(testset,testtarget,model):
  
    predicts = model.predict(testset)
    erro = metrics.mean_squared_error(testtarget,predicts)

    return (predicts,erro)        

dimension = 13
datafile = 'downjonesv2.txt'

(data_set, data_target, val_set, val_target, pred_set, pred_target) = processdata(datafile , dimension)

#MLP

mlp_reg = createmodel(data_set, data_target, val_set, val_target, 'MLP')
(pred_mlp, erro_mlp) = predict(pred_set, pred_target, mlp_reg)

print 'MlP Erro: %.4f' %erro_mlp
"""

#Plotagem
x = range(len(pred_target))


plt.plot(x, pred_target, 'r--', label = 'Real')
plt.plot(x, pred_mlp, label = 'MLP predicted')
plt.legend()
plt.figure

"""