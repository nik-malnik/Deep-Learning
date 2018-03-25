import model_run
import numpy as np
import copy
import random
from matplotlib import pyplot as plt

def sigmoid(z):
	return 1.0/(1.0 + np.exp(-z))

def h_cond_sample(x_sample, W, a, b):
	p = sigmoid(np.dot(W.T,x_sample) + a)
	return np.reshape(np.random.binomial(1,p[:,0]), (-1,1))

def x_cond_sample(h_sample, W, a, b):
	p = sigmoid(np.dot(W,h_sample) + b)
	return np.reshape(np.random.binomial(1,p[:,0]), (-1,1))

def CD(x_sample, W, a, b, K, lr, update_params = 1):
	h_sample = h_cond_sample(x_sample, W, a, b)
	h_sample_new = copy.deepcopy(h_sample)
	for k in range(K):
		x_sample_new = x_cond_sample(h_sample_new, W, a, b)
		h_sample_new = h_cond_sample(x_sample_new, W, a, b)
	
	if update_params:
		W = W + lr*(np.dot(x_sample,h_sample.T) - np.dot(x_sample_new,h_sample_new.T))
		a = a + lr*(h_sample - h_sample_new)
		b = b + lr*(x_sample - x_sample_new)
		return [W,a,b]
	else:
		return x_sample_new

def cross_entropy(X, W, a, b):
	H = sigmoid(np.dot(W.T,X) + a)
	X_new = sigmoid(np.dot(W,H) + b)
	
	recon_error = (1.0/X.shape[1])*np.sum(np.square(X-X_new))
	cross_entropy_error = (1.0/X.shape[1])*np.sum((X*np.log(X_new)) + ((1.0-X)*np.log(1 - X_new)))
	return recon_error, cross_entropy_error
	
def main(hidden_units= 100, epochs = 1, LR = 0.05, K = 1):
	
	[xtrain, ytrain, xval, yval, xtest, ytest] = model_run.load_mnist()
	performance_trend = {'train_cross_entropy': [], 'validation_cross_entropy': [], 'train_recon_error': [], 'validation_recon_error': []}
	W = np.random.normal(0.0,0.1,size=(xtrain.shape[0],hidden_units))
	a = np.random.normal(0.0,0.1,size=(hidden_units,1))
	b = np.random.normal(0.0,0.1,size=(xtrain.shape[0],1))
	indices = range(xtrain.shape[1])
	
	for epoch in range(epochs):
		random.shuffle(indices)
		
		if epoch < 10:
			lr = LR
		elif epoch < 100:
			lr = LR/np.sqrt(epoch - 8.0)
		else:
			lr = LR/(epoch - 90)
		
		for i in indices:
			
			[W,a,b] = CD(xtrain[:,i].reshape(-1,1),W,a,b,K,lr)
		recon_error, cross_entropy_error = cross_entropy(xtrain, W, a, b)
		performance_trend['train_cross_entropy'].append(cross_entropy_error)
		performance_trend['train_recon_error'].append(recon_error)
		recon_error, cross_entropy_error = cross_entropy(xval, W, a, b)
		performance_trend['validation_cross_entropy'].append(cross_entropy_error)
		performance_trend['validation_recon_error'].append(recon_error)
		
		print "\nCompleted epoch:" + str(epoch)
		print 'Cross Entropy Training ' + str(performance_trend['train_cross_entropy'][-1]) + ', Validation:' + str(performance_trend['validation_cross_entropy'][-1])
	  
	plt.plot(range(epochs),performance_trend['train_cross_entropy'], c = 'b')
	plt.plot(range(epochs),performance_trend['validation_cross_entropy'], c = 'r')
	print cross_entropy(xtest, W, a ,b)
	
	return [W,a,b, performance_trend]

def plot_weights(W):
	
	f, plots = plt.subplots(10, 10)
	for i in range(100):
		plots[i/10,i%10].imshow(W[:,i].reshape(28,28), cmap='gray')
	plt.show()

def generate_digits(W,a,b):
	
	[xtrain, ytrain, xval, yval, xtest, ytest] = model_run.load_mnist()
	indices = range(3000)
	random.shuffle(indices)
	f, plots = plt.subplots(10, 10)
	for i in range(100):
		x_sample = xtest[:,indices[i]:indices[i]+1]*np.random.binomial(1,0.85,size=(784,1))
		x_sample_new = CD(x_sample, W, a, b, 1000, 0.01, update_params=0)
		plots[i/10,i%10].imshow(x_sample_new.reshape(28,28), cmap='gray')
	plt.show()

def k_experiment(epochs = 1):
	results = []
	for K in [1,5,10,20]:
		results.append({'K' : K, 'res': main(epochs = epochs, K = K)})
	return results
	
	for result in results:
		plt.plot(range(20),result['res'][3]['train_cross_entropy'], c = 'b', lw = 0.1*result['K'])
		plt.plot(range(20),result['res'][3]['validation_cross_entropy'], c = 'r', lw = 0.1*result['K'])

def hidden_unit_experiment(epochs = 1):
	results = []
	for hidden_units in [50,100,200,500]:
		results.append({'hidden_units' : hidden_units, 'res': main(hidden_units = hidden_units, epochs = epochs)})
	return results	

main()
