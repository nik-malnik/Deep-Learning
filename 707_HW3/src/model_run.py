import numpy as np
import network_configure
import pickle
import copy
import random
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import language_model as LM
  
def network():

  layers = {}
  layers[1] = {'type':'DATA','height':8000,'channel':1,'batch_size':64}
  layers[2] = {'type':'Embed','num':48}
  layers[3] = {'type':'Tanh'}
  layers[3] = {'type':'IP','num':128}
  layers[4] = {'type':'Tanh'}
  layers[5] = {'type':'LOSS','num':8000}
  
  
  return layers

def main(epochs = 10,l_rate = 0.1, layers = network()):
  
  xtrain, ytrain, xval, yval = LM.create_train_val_dataset()

  batch_size = layers[1]['batch_size']
  performance_trend = {'train_cross_entropy': [], 'validation_cross_entropy': [], 'train_success_rate': [], 'validation_success_rate': []}

  params = network_configure.init_network(layers)

  indices = range(xtrain.shape[1])
  for epoch in range(epochs):
    random.shuffle(indices)
    for step in range(xtrain.shape[1]/batch_size):
      idx = indices[step*batch_size:(step+1)*batch_size]
      data = (np.arange(8000) == (xtrain[:, idx]+1)[...,None]-1).astype(int)
      data = data.reshape(data.shape[2],data.shape[0],data.shape[1])
      [obj, success_rate, param_grad] = network_configure.network(params,layers,data,ytrain[idx])
      params = network_configure.sgd_momentum(l_rate,params,param_grad)
      if step%100 == 0:
		  print str(step) + ":" + str(obj) + ":" + str(success_rate*100) + "%"
    
    if (epoch+1)%10 == 0:
      l_rate = l_rate/1.5
    if (epoch+1)%25 == 0:
      batch_size = batch_size*2
    
    #layers[1]['batch_size'] = xval.shape[1]
    #temp = network_configure.network(params, layers, xval, yval)
    #performance_trend['validation_cross_entropy'].append(temp[0])
    #performance_trend['validation_success_rate'].append(temp[1])
    #layers[1]['batch_size'] = xtrain.shape[1]
    #temp = network_configure.network(params, layers, xtrain, ytrain)
    #performance_trend['train_cross_entropy'].append(temp[0])
    #performance_trend['train_success_rate'].append(temp[1])
    #layers[1]['batch_size'] = batch_size
    #print "\nCompleted epoch:" + str(epoch)
    #print '\nTSR: ' + str(performance_trend['train_success_rate'][-1]) + ', VSR:' + str(performance_trend['validation_success_rate'][-1]) + ', TCE: ' + str(performance_trend['train_cross_entropy'][-1]) + ', VCE:' + str(performance_trend['validation_cross_entropy'][-1])
      
  #plt.plot(range(epochs),performance_trend['train_success_rate'])
  #plt.plot(range(epochs),performance_trend['validation_success_rate'])
  
  return performance_trend, params

xtrain, ytrain, xval, yval = LM.create_train_val_dataset()
main()
