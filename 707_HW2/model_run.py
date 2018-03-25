import numpy as np
import network_configure
import pickle
import copy
import random
from PIL import Image
import matplotlib.pyplot as plt

def load_mnist(fullset=True):

  fs = ['digitstrain.txt', 'digitsvalid.txt', 'digitstest.txt'] 
  lens = [3000,1000,3000] 
  y = [[],[],[]]
  x = [[],[],[]]
  for i in range(3):
	  temp_x = []
	  f = open(fs[i],'rb')
	  for line_num in range(lens[i]):
		  line = f.readline().split(',')
		  x[i].append([float(t) for t in line[:784]])
		  y[i].append(int(line[-1].strip()))
  
  xtrain = np.vstack(x[0])	
  ytrain = np.vstack(y[0])	  
  xvalidate = np.vstack(x[1])	
  yvalidate = np.vstack(y[1])	
  xtest = np.vstack(x[2])	
  ytest = np.vstack(y[2])	

  return [xtrain.T, ytrain[:,0], xvalidate.T, yvalidate[:,0], xtest.T, ytest[:,0]]
  
def network():

  layers = {}
  layers[1] = {'type':'DATA','height':784,'channel':1,'batch_size':1}
  layers[2] = {'type':'IP','num':500}
  layers[3] = {'type':'Sigmoid'}
  layers[4] = {'type':'IP','num':784}
  layers[5] = {'type':'Sigmoid'}
  layers[6] = {'type':'autoEnc','num' : 10}
  
  return layers


def main(epochs = 10, mu = 0,l_rate = 0.01, weight_decay = 0, save = 1, model_name = 'network_params.mat', layers = network(), reuse_weights = False, weight_type = 'rbm', addnoise = False):
  
  [xtrain, ytrain, xval, yval, xtest, ytest] = load_mnist()

  batch_size = layers[1]['batch_size']
  performance_trend = {'train_cross_entropy': [], 'validation_cross_entropy': [], 'train_error_rate': [], 'validation_error_rate': []}
  
  params = network_configure.init_network(layers,reuse_weights = reuse_weights, weight_type = weight_type)
  parameter_history = copy.deepcopy(params)
  for l_idx in range(1, len(layers)):
    parameter_history[l_idx]['w'] = np.zeros(parameter_history[l_idx]['w'].shape)
    parameter_history[l_idx]['b'] = np.zeros(parameter_history[l_idx]['b'].shape)
  
  indices = range(xtrain.shape[1])
  for epoch in range(epochs):
    random.shuffle(indices)
    for step in range(xtrain.shape[1]/batch_size):
      idx = indices[step*batch_size:(step+1)*batch_size]
      [obj, success_rate, param_grad] = network_configure.network(params,layers,xtrain[:, idx],ytrain[idx], addnoise = addnoise)
      params, parameter_history = network_configure.sgd_momentum(l_rate,mu,weight_decay,params,parameter_history,param_grad)

    layers[1]['batch_size'] = xval.shape[1]
    performance_trend['validation_cross_entropy'].append(1.0 - network_configure.network(params, layers, xval, yval)[0])
    performance_trend['validation_error_rate'].append(network_configure.network(params, layers, xval, yval)[1])
    layers[1]['batch_size'] = xtrain.shape[1]
    performance_trend['train_cross_entropy'].append(1.0 - network_configure.network(params, layers, xtrain, ytrain)[0])
    performance_trend['train_error_rate'].append(network_configure.network(params, layers, xtrain, ytrain)[1])
    layers[1]['batch_size'] = batch_size
    print "\nCompleted epoch:" + str(epoch)
    print 'Training Error: ' + str(performance_trend['train_error_rate'][-1])
    print 'Validation Error:' + str(performance_trend['validation_error_rate'][-1])
  
  plt.plot(range(epochs),performance_trend['train_error_rate'], c='b')
  plt.plot(range(epochs),performance_trend['validation_error_rate'],c='r')
  layers[1]['batch_size'] = xtest.shape[1]
  print (network_configure.network(params, layers, xtest, ytest)[1])
  
  if save == 1 :
    pickle.dump(params, open(model_name, 'wb'))
  
  return performance_trend
      
def Plots( model_name = 'network_params.mat' ):
	
	params = pickle.load(open(model_name,'rb'))	
	f, plots = plt.subplots(10, 10)
	for i in range(100):
		plots[i/10,i%10].imshow(params[1]['w'][:,i].reshape(28,28), cmap='gray')
	
	plt.show()

def plot_reconstruction( model_name = 'network_params.mat'):
	params = pickle.load(open(model_name,'rb'))	
	layers = network()
	[xtrain, ytrain, xval, yval, xtest, ytest] = load_mnist()
	x_new = network_configure.network(params, layers, xtest[:,range(0,3000,300)], ytest[range(0,3000,120)], reconstruction = True)
	f, plots = plt.subplots(10, 2)
	for i in range(10):
		plots[i,0].imshow(xtest[:,range(0,3000,300)][:,i].reshape(28,28), cmap='gray')
		plots[i,1].imshow(x_new[:,i].reshape(28,28), cmap='gray')
	plt.show()
  
def hidden_unit_experiment(epochs = 1):
	results = []
	layers = network()
	for hidden_units in [50,100,200,500]:
		layers[2]['num'] = hidden_units
		results.append({'hidden_units' : hidden_units, 'res': main(layers = layers, l_rate = 0.03, epochs = epochs)})
	return results	

main()
