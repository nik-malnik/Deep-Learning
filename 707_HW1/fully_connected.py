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

