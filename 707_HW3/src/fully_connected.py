import numpy as np
import math
import scipy.io
import copy

def inner_product_forward(input, layer, param):

  output = {'height':1, 'channel':layer['num'], 'batch_size':input['batch_size']}
  output['data'] = np.zeros((layer['num'], input['batch_size']))
  output['data'] = ((np.matmul(param['w'].T,input['data'])).T + param['b']).T

  return output


def inner_product_backward(output, input, layer, param):

  param_grad = {}
  param_grad['b'] = np.zeros(param['b'].shape)
  param_grad['w'] = np.zeros(param['w'].shape)
  input_od = np.zeros(input['data'].shape)
  
  batch_size = input['batch_size']
  input_od = np.dot(param['w'],output['diff'])
  param_grad['b'] = np.sum(output['diff'],axis=1)
  for n in range(batch_size):
    temp1 = input['data'][:,n]
    temp2 = output['diff'][:,n]
    param_grad['w'] += np.dot(temp1.reshape(temp1.shape[0],1),temp2.reshape((1,temp2.shape[0])))

  return param_grad, input_od

def embedding_forward(input, layer, param):

  output = {'height':1, 'channel':layer['num'], 'batch_size':input['batch_size']}
  output['data'] = np.zeros((layer['num']/3,3, input['batch_size']))
  output['data'] = np.tensordot(param['w'], input['data'],axes=[[0],[0]])
  output['data'] = output['data'].reshape(output['data'].shape[0]*output['data'].shape[1],output['data'].shape[2])

  return output

def embedding_backward(output, input, layer, param):

  param_grad = {}
  param_grad['b'] = np.zeros(param['b'].shape)
  param_grad['w'] = np.zeros(param['w'].shape)
  input_od = np.zeros(input['data'].shape)
  
  batch_size = input['batch_size']
  output['diff'] = output['diff'].reshape(output['diff'].shape[0]/3,3,output['diff'].shape[1])
  for n in range(batch_size):
    temp1 = input['data'][:,:,n]
    temp2 = output['diff'][:,:,n]
    param_grad['w'] += np.matmul(temp1,temp2.T)

  return param_grad, input_od
