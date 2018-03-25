import numpy as np
import network_configure
import pickle
import copy
import random
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import sys
from scipy.interpolate import interp1d

from keras.layers import Input, Dense, Reshape, Dropout, Activation, Flatten, LSTM, SimpleRNN, Conv2D, Concatenate, Embedding
from keras.models import Model, Sequential
from keras.optimizers import SGD, Adadelta
import keras

sys.setrecursionlimit(10000)
	
def load_data(f):
  X = []
  with open(f) as fobj:
    for line in fobj:
        X.append(['START'] + [x.lower() for x in line.split(' ')] + ['END'])	
  return X
  
def vocabulary(X,N = 8000):
	dict = Counter(sum(X,[]))
	df = pd.DataFrame.from_dict(dict, orient='index').reset_index()
	df = df.sort(columns=0,ascending=False)[:N-1]
	vocab = list(df['index']) + ['UNK']
	return vocab

def replace_by_id(X,vocab):
	N = len(vocab)	
	for i in range(len(X)):
		X[i] =  map(lambda x: vocab.index(x) if x in vocab else N-1, X[i])
	return X
  
def find_four_grams(X,vocab, analyse = False):
	four_grams = []
	for i in range(len(X)):
		for j in range(3,len(X[i])):
			four_grams.append((X[i][j-3],X[i][j-2],X[i][j-1],X[i][j]))
	
	if analyse:
		dict = Counter(four_grams)
		df = pd.DataFrame.from_dict(dict, orient='index').reset_index()
		df = df.sort(columns=0,ascending=False)
		plt.hist(np.log(df[0]),20)
		top_four_grams = list(df[:50]['index'])
		for i in range(50):
			str = ''
			for j in range(4):
				str = str + vocab[top_four_grams[i][j]] + ' '
			print str + '\n'
	
	return four_grams

def create_train_val_dataset():
	Xtrain = load_data('train.txt')
	Xval = load_data('val.txt')
  
	vocab = vocabulary(Xtrain)
	X = replace_by_id(Xtrain,vocab)
	X = find_four_grams(X,vocab, analyse = False)
	xtrain = np.array([[x[0],x[1],x[2]] for x in X]).T
	ytrain = np.array([x[3] for x in X])
  
	X = replace_by_id(Xval,vocab)
	X = find_four_grams(X,vocab, analyse = False)
	xval = np.array([[x[0],x[1],x[2]] for x in X]).T
	yval = np.array([x[3] for x in X])
	
	return xtrain, ytrain, xval, yval	

def bonus_approach():
	xtrain, ytrain, xval, yval = create_train_val_dataset()
	
	history = NN_finetune(xtrain,ytrain, xval, yval, E = 32)
	pickle.dump(save_stats(history), open( 'NN3_E32.pckl','wb'))
	history = NN_finetune(xtrain,ytrain, xval, yval, E = 64)
	pickle.dump(save_stats(history), open( 'NN3_E64.pckl','wb'))
	history = NN_finetune(xtrain,ytrain, xval, yval, E = 128)
	pickle.dump(save_stats(history), open( 'NN3_E128.pckl','wb'))


def NN_finetune(xtrain,ytrain, xval, yval, E = 16):

	epochs = [6,4,3,2,2,10]
	excludes = [[0,1,2,3,4,5,6,7,8,9,7999],[0,1,2,3,4,5,6,7999], [0,1,2,3,7999], [0,1,7999], [0], []]
	load_model = ''
	history = []
	for i in range(len(epochs)):
	  t = np.in1d(ytrain, excludes[i])
	  idx = np.where(t==False)
	  print "\nTotal training samples:" + str(len(idx[0]))
	  ytrain_new = np.zeros((8000,ytrain[idx].shape[0]))
	  ytrain_new[ytrain[idx], np.arange(ytrain[idx].shape[0])] = 1
	  yval_new = np.zeros((8000,yval.shape[0]))
	  yval_new[yval, np.arange(yval.shape[0])] = 1
	  model4, h4 = NN3(xtrain.T[idx], ytrain_new.T, xval.T, yval_new.T, nb_epoch = epochs[i], embedding_layer = E, load_model = load_model )
	  history.append(h4)
	  #print np.unique(np.argmax(model4.predict(xval.T),axis=1),return_counts = True)
	  load_model = 'NN3_' + str(epochs[i]) + '_64_' + str(E) + '_.h5' 
	return history

def NN3(xtrain, ytrain, xval, yval, nb_epoch = 1, batch_size = 64, embedding_layer = 16, load_model = '' ):

	x_input = Input(shape=(3,))
	x = Embedding(8000, embedding_layer, input_length=3)(x_input)
	x = SimpleRNN(128,  input_shape=(3, embedding_layer),  return_sequences=False)(x)
	x = Dense(8000,activation = 'softmax')(x)
	model = Model(x_input,x)
	if load_model != '':
		model.load_weights(load_model)
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['acc'])
	history = model.fit(xtrain, ytrain, validation_data = (xval, yval), epochs=nb_epoch, batch_size=batch_size)
	name = 'NN3_' + str(nb_epoch) + '_' +  str(batch_size) + '_'  +  str(embedding_layer) + '_'  + '.h5'
	model.save( name )
	#pickle.dump(history, open( name + '.pckl','wb'))
	print "Saved model:" + name
	
	return model, history

def NN4(xtrain, ytrain, nb_epoch = 1, batch_size = 1 ):
	
	x_input = Input(shape=(3,))
	x = Reshape((3,1))(x_input)
	x = SimpleRNN(128,  input_shape=(3, 1),  return_sequences=False)(x)
	x = Dense(8000,activation = 'softmax')(x)
	model = Model(x_input,x)
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['acc'])
	history = model.fit(xtrain, ytrain, epochs=nb_epoch, batch_size=batch_size)
	return model, history

def print_sentence(t,vocab):
	st = ''
	for i in range(len(t)):
		st = st + vocab[t[i]] + ' '
	print repr(st)

def generate_sentence(model):
	starts = [[311,5,42], [45,20,1], [1051, 5, 233], [45,56,6]]
	for start in starts: 
		for i in range(8):
			start.append(np.argmax(model.predict([np.array([start[-3]]),np.array([start[-2]]),np.array([start[-1]])]),axis=1)[0])
		print_sentence(start,vocab)
		print '\n\n'

def save_stats(history):
	hist = []
	for his in history:
		hist.append(his.history)
	return hist

def calculate_stats(hist):
	stats = {'acc': [], 'ce' : [], 'plxt': [], 'val_acc': [], 'val_ce': [], 'val_plxt': [] }
	for his in hist:
		stats['acc'] = stats['acc'] + his['acc']
		stats['ce'] += his['loss']
		stats['val_acc'] += his['val_acc']
		stats['val_ce'] += his['val_loss']
		stats['plxt'] += [np.power(2,t) for t in his['loss']]
		stats['val_plxt'] += [np.power(2,t) for t in his['val_loss']]
	return stats


bonus_approach()


