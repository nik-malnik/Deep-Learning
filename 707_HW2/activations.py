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
