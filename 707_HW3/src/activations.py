import numpy as np
import math
import scipy.io
import copy

def relu_forward(input, layer):
	
  output = {'height':input['height'], 'channel': input['channel'], 'batch_size' : input['batch_size']}
  output['data'] = np.maximum(input['data'],np.zeros(input['data'].shape))
  return output

def relu_backward(output, input, layer):
  input_od = np.zeros(input['data'].shape)
  batch_size = input['batch_size']
  input_od = (output['data'] > 0)*output['diff']

  return input_od

def sigmoid_forward(input, layer):
  
  output = {'height':input['height'], 'channel': input['channel'], 'batch_size' : input['batch_size']}
  output['data'] = (1.0/(1.0 + np.exp(-1*input['data'])))
  return output

def sigmoid_backward(output, input, layer):
  input_od = np.zeros(input['data'].shape)
  batch_size = input['batch_size']
  input_od = output['data']*(1.0 - output['data'])*output['diff']

  return input_od

def tanh_forward(input, layer):
  
  output = {'height':input['height'], 'channel': input['channel'], 'batch_size' : input['batch_size']}
  output['data'] = ((1.0 - np.exp(-2*input['data']))/(1.0 + np.exp(-2*input['data'])))
  return output

def tanh_backward(output, input, layer):
  input_od = np.zeros(input['data'].shape)
  batch_size = input['batch_size']
  input_od = (1.0 - np.power(output['data'],2))*output['diff']

  return input_od

def batch_normalization_forward(input, layer, param):
  
  output = {'height':input['height'], 'channel': input['channel'], 'batch_size' : input['batch_size']}
  epsilon = 0.0001
  mu = np.mean(input['data'],axis=1)
  var = np.var(input['data'],axis=1)
  output['data'] = ((((input['data'].T - mu)/np.sqrt(var + epsilon)))*param['w'] + param['b']).T
  return output

def batch_normalization_backward(output, input, layer, param):

  param_grad = {}
  epsilon = 0.0001
  mu = np.mean(input['data'],axis=1)
  var = np.var(input['data'],axis=1)
  param_grad['b'] = np.sum(output['diff'],axis=1)
  param_grad['w'] = np.sum(output['diff']*((output['data'].T - param['b'])/param['w']).T, axis=1)
  input_od = (param_grad['w']*np.power(var+epsilon,-0.5)*(input['batch_size']*output['diff'].T - np.sum(output['diff'],axis=1) - (input['data'].T - mu)*np.power(var + epsilon,-1)*np.sum(output['diff'].T*(input['data'].T - mu),axis=0))).T
  return param_grad, input_od
