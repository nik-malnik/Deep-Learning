import numpy as np
import math
import scipy.io
import copy
import activations
import fully_connected
import pickle

# Parts of this file were reused from ML 601 Assignment 7 in Spring 2017.

def init_network(layers, reuse_weights = False, weight_type = 'rbm'):

  params = {}

  h = layers[1]['height']
  c = 1

  for i in range(2, len(layers)+1):
    params[i-1] = {}
    if layers[i]['type'] == 'IP':
		
      if reuse_weights:
		  print "Loading pre-trained weights: " + weight_type 
		  if weight_type == 'rbm':
			  [W, a ,b] = pickle.load(open('rbm','rb'))
			  params[i-1]['w'] = W
			  params[i-1]['b'] = a[:,0]
		  elif weight_type == 'ae':
			  params_transfer = pickle.load(open('ae','rb'))
			  params[i-1]['w'] = params_transfer[1]['w']
			  params[i-1]['b'] = params_transfer[1]['b']
		  elif weight_type == 'ae_dropout':
			  params_transfer = pickle.load(open('ae_dropout','rb'))
			  params[i-1]['w'] = params_transfer[1]['w']
			  params[i-1]['b'] = params_transfer[1]['b']		
			    
      else:
        params[i-1]['w'] = np.sqrt(3./(h*c))*np.random.randn(h*c, layers[i]['num'])
        params[i-1]['b'] = np.zeros(layers[i]['num'])
      
      h = 1
      c = layers[i]['num']
    
    elif layers[i]['type'] == 'RELU':
      params[i-1]['w'] = np.array([])
      params[i-1]['b'] = np.array([])
    elif layers[i]['type'] == 'Sigmoid':
      params[i-1]['w'] = np.array([])
      params[i-1]['b'] = np.array([])
    elif layers[i]['type'] == 'Tanh':
      params[i-1]['w'] = np.array([])
      params[i-1]['b'] = np.array([])
    elif layers[i]['type'] == 'LOSS':
      params[i-1]['w'] = 2*np.sqrt(3./(h*c))*np.random.rand(h*c, layers[i]['num']-1) - np.sqrt(3./(h*c))
      params[i-1]['b'] = np.zeros(layers[i]['num'] - 1)
      h = 1
      c = layers[i]['num']
    elif layers[i]['type'] == 'autoEnc':
      params[i-1]['w'] = np.array([])
      params[i-1]['b'] = np.array([])
  return params


def network(params, layers, data, labels, reconstruction = False, addnoise = False):

  l = len(layers)
  batch_size = layers[1]['batch_size']

  param_grad = {}
  cp = {}
  output = {}
  
  data_orig = copy.deepcopy(data)
  if addnoise:
    noise = np.random.binomial(1,0.75,size=data.shape)
    data = data*noise  
  
  output[1] = {'data': data, 'height':layers[1]['height'], 'channel':layers[1]['channel'], 'batch_size': layers[1]['batch_size'], 'diff' : 0 }
  
  for i in range(2, l+1):
    if layers[i]['type'] == 'IP':
      output[i] = fully_connected.inner_product_forward(output[i-1], layers[i], params[i-1])
    elif layers[i]['type'] == 'RELU':
      output[i] = activations.relu_forward(output[i-1], layers[i])
    elif layers[i]['type'] == 'Sigmoid':
      output[i] = activations.sigmoid_forward(output[i-1], layers[i])
    elif layers[i]['type'] == 'Tanh':
      output[i] = activations.tanh_forward(output[i-1], layers[i])
    elif layers[i]['type'] == 'LOSS':
      [obj, grad_w,grad_b, input_back_deriv, success_rate] = loss_func(params[i-1]['w'],params[i-1]['b'],output[i-1]['data'],labels,layers[i]['num'], 1)
      param_grad[i-1] = {'w': grad_w/batch_size, 'b' : grad_b/batch_size}
    elif layers[i]['type'] == 'autoEnc':
      [obj, input_back_deriv, success_rate] = autoEnc_loss(output[i-1]['data'],data_orig)
      param_grad[i-1] = {'w': 0.0, 'b': 0.0}
  
  if reconstruction:
	  return output[i-1]['data']
  
  for i in range(l-1,1,-1):
    param_grad[i-1] = {}
    param_grad[i-1]['w'] = np.array([])
    param_grad[i-1]['b'] = np.array([])
    if layers[i]['type'] == 'IP':
      output[i]['diff'] = input_back_deriv
      param_grad[i-1], input_back_deriv = fully_connected.inner_product_backward(output[i],output[i-1],layers[i],params[i-1])
    elif layers[i]['type'] == 'RELU':
      output[i]['diff'] = input_back_deriv
      input_back_deriv = activations.relu_backward(output[i], output[i-1], layers[i])
      param_grad[i-1]['w'] = np.array([])
      param_grad[i-1]['b'] = np.array([])
    elif layers[i]['type'] == 'Sigmoid':
      output[i]['diff'] = input_back_deriv
      input_back_deriv = activations.sigmoid_backward(output[i], output[i-1], layers[i])
      param_grad[i-1]['w'] = np.array([])
      param_grad[i-1]['b'] = np.array([])
    elif layers[i]['type'] == 'Tanh':
      output[i]['diff'] = input_back_deriv
      input_back_deriv = activations.tanh_backward(output[i], output[i-1], layers[i])
      param_grad[i-1]['w'] = np.array([])
      param_grad[i-1]['b'] = np.array([])

  return (obj/batch_size),success_rate, param_grad

def autoEnc_loss(X_new, X):
  cross_entropy_loss = (1.0/X.shape[0])*np.sum((X*np.log(X_new)) + ((1.0-X)*np.log(1 - X_new)))
  reconstruction_loss = (1.0/X.shape[1])*np.sum(np.square(X - X_new))
  back_deriv = -(X/X_new) + ((1-X)/(1-X_new))
  return [cross_entropy_loss, back_deriv, reconstruction_loss]

def loss_func(w, b, X, y_true, labels, prediction):
	
  prob = (np.dot(X.T,w) + b).T
  prob = np.vstack([prob,np.zeros(X.shape[1])])
  prob = np.exp(prob - np.max(prob,axis=0))
  prob = np.divide(prob,np.sum(prob,axis=0))
  obj = -np.sum(np.log(prob[y_true,np.arange(X.shape[1])]))
  
  if prediction == 1:
    indices = np.argmax(prob, axis=0)
    percent = len(np.where(y_true == indices)[0]) / float(len(y_true))
  else:
    percent = 0
  
  back_deriv = prob - np.eye(labels)[y_true].T
  grad_w = np.dot(back_deriv,X.T)[0:-1,:].T
  grad_b = np.sum(back_deriv, axis=1)[0:-1]
  back_deriv = np.dot(w,back_deriv[0:-1, :])

  return obj, grad_w,grad_b, back_deriv, percent

def sgd_momentum(l_rate, mu, weight_decay, params, parameter_history, param_grad):

  params_ = copy.deepcopy(params)
  parameter_history_ = copy.deepcopy(parameter_history)
  for i in range(1,len(params_)+1):
    parameter_history_[i]['w'] = mu*parameter_history_[i]['w'] + param_grad[i]['w'] + weight_decay*params[i]['w']
    parameter_history_[i]['b'] = mu*parameter_history_[i]['b'] + param_grad[i]['b']
    params_[i]['w'] = params[i]['w'] -  l_rate*parameter_history_[i]['w']
    params_[i]['b'] = params[i]['b'] -  l_rate*parameter_history_[i]['b']

  return params_, parameter_history_
