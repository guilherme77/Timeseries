# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 15:07:57 2019

@author: Guilherme
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

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
         
        #Treinamento  80
        trainset = laggedata.iloc[dimension:trainindex,1:dimension+1]
        traintarget = laggedata.iloc[dimension:trainindex,0]
        
        #Validação 20
        valset = laggedata.iloc[trainindex:valindex,1:dimension+1]
        valtarget = laggedata.iloc[trainindex:valindex,0]
        
        #Teste 20
        testset = laggedata.iloc[valindex:len(data),1:dimension+1]
        testtarget =  laggedata.iloc[valindex:len(data),0]
        
        return (trainset,traintarget,valset,valtarget,testset,testtarget)

def predict(testset,testtarget,model):
  
    predicts = model.predict(testset)
    erro = metrics.mean_squared_error(testtarget,predicts)

    return (predicts,erro)

def createmodelAR(trainset,traintarget,dimension):    
    trainset[dimension+1]=1 
    coefs = np.linalg.pinv(trainset).dot(traintarget)
    #print coefs
    return coefs

def predictAR(coefs,testset,testtarget,dimension):
    
    testset[dimension+1]=1
    predicts = testset.dot(coefs)
    erro = metrics.mean_squared_error(testtarget, predicts)

    return (predicts, erro)

dimension =13
datafile = 'downjonesv2.txt'

#AutoRegressivo

(train_set, train_target, test_set, test_target) = processdata(datafile, dimension, valid=False)
coefs = createmodelAR(train_set, train_target, dimension)
(pred_AR, erro_AR) = predictAR(coefs, test_set, test_target, dimension) 

print 'AR  Erro: %.4f' %erro_AR
#print 'AR  pred: %s' %pred_AR
"""
plt.plot(test_target, 'r--', label = 'Real')
plt.plot(pred_AR, label = 'AR predicted')
plt.legend()
plt.figure
"""
