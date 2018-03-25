import numpy as np
import math
import scipy.io
import copy
import activations
import fully_connected

# Parts of this file were reused from ML 601 Assignment 7 in Spring 2017.

def init_network(layers):

  params = {}

  h = layers[1]['height']
  c = 1

  for i in range(2, len(layers)+1):
    params[i-1] = {}
    if layers[i]['type'] == 'IP':
      params[i-1]['w'] = np.sqrt(3./(h*c))*np.random.randn(h*c, layers[i]['num'])
      params[i-1]['b'] = np.zeros(layers[i]['num'])
      h = 1
      c = layers[i]['num']
    elif layers[i]['type'] == 'batch_norm':
      params[i-1]['w'] = np.ones(c)
      params[i-1]['b'] = 0.01*np.ones(c)
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
  return params


def network(params, layers, data, labels):

  l = len(layers)
  batch_size = layers[1]['batch_size']

  param_grad = {}
  cp = {}
  output = {}
  output[1] = {'data': data, 'height':layers[1]['height'], 'channel':layers[1]['channel'], 'batch_size': layers[1]['batch_size'], 'diff' : 0 }
  
  for i in range(2, l):
    if layers[i]['type'] == 'IP':
      output[i] = fully_connected.inner_product_forward(output[i-1], layers[i], params[i-1])
    elif layers[i]['type'] == 'RELU':
      output[i] = activations.relu_forward(output[i-1], layers[i])
    elif layers[i]['type'] == 'Sigmoid':
      output[i] = activations.sigmoid_forward(output[i-1], layers[i])
    elif layers[i]['type'] == 'Tanh':
      output[i] = activations.tanh_forward(output[i-1], layers[i])
    elif layers[i]['type'] == 'batch_norm':
      output[i] = activations.batch_normalization_forward(output[i-1], layers[i], params[i-1])
  i = l
  [obj, grad_w,grad_b, input_back_deriv, success_rate] = loss_func(params[i-1]['w'],params[i-1]['b'],output[i-1]['data'],labels,layers[i]['num'], 1)

  param_grad[i-1] = {'w': grad_w/batch_size, 'b' : grad_b/batch_size}


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
    elif layers[i]['type'] == 'batch_norm':
      output[i]['diff'] = input_back_deriv
      param_grad[i-1], input_back_deriv = activations.batch_normalization_backward(output[i],output[i-1],layers[i],params[i-1])
    param_grad[i-1]['w'] = param_grad[i-1]['w'] / batch_size
    param_grad[i-1]['b'] = param_grad[i-1]['b'] / batch_size

  return (obj/batch_size),success_rate, param_grad

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
