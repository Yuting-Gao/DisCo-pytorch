
import pandas as pd
import numpy as np

NUM_LABEL=10

def MI_cal_v2(label_matrix, layer_T, NUM_TEST_MASK):
	'''
	Inputs:
	- size_of_test: (N,) how many test samples have be given. since every input is different
	  we only care the number.
	-  label_matrix: (N,C)  the label_matrix created by creat_label_matrix.py.
	-  layer_T:  (N,H) H is the size of hidden layer
	- NUM_TEST_MASK: sample size
	Outputs:
	- MI_XT : the mutual information I(X,T)
	- MI_TY : the mutual information I(T,Y)
	'''
	MI_XT=0
	MI_TY=0
	layer_T = np.exp(layer_T - np.max(layer_T,axis=1,keepdims=True))
	layer_T /= np.sum( layer_T,axis=1,keepdims=True)
	layer_T = Discretize_v2(layer_T)
	XT_matrix = np.zeros((NUM_TEST_MASK,NUM_TEST_MASK))
	Non_repeat=[]
	mark_list=[]
	for i in range(NUM_TEST_MASK):
		pre_mark_size = len(mark_list)
		if i==0:
			Non_repeat.append(i)
			mark_list.append(i)
			XT_matrix[i,i]=1
		else:
			for j in range(len(Non_repeat)):
				if (layer_T[i] ==layer_T[ Non_repeat[j] ]).all():
					mark_list.append(Non_repeat[j])
					XT_matrix[i,Non_repeat[j]]=1
					break
		if pre_mark_size == len(mark_list):
			Non_repeat.append(Non_repeat[-1]+1)
			mark_list.append(Non_repeat[-1])
			XT_matrix[i,Non_repeat[-1]]=1

	XT_matrix = np.delete(XT_matrix,range(len(Non_repeat),NUM_TEST_MASK),axis=1)
	P_layer_T = np.sum(XT_matrix,axis=0)/float(NUM_TEST_MASK)
	P_sample_x = 1/float(NUM_TEST_MASK)
	for i in range(NUM_TEST_MASK):
		MI_XT+=P_sample_x*np.log2(1.0/P_layer_T[mark_list[i]])


	TY_matrix = np.zeros((len(Non_repeat),NUM_LABEL))
	mark_list = np.array(mark_list)
	for i in range(len(Non_repeat)):
		TY_matrix[i,:] = np.sum(label_matrix[  np.where(mark_list==i)[0]  , : ] ,axis=0 )
	TY_matrix = TY_matrix/NUM_TEST_MASK
	P_layer_T_for_Y = np.sum(TY_matrix,axis=1)
	P_Y_for_layer_T = np.sum(TY_matrix,axis=0)
	for i in range(TY_matrix.shape[0]):
		for j in range(TY_matrix.shape[1]):
			if TY_matrix[i,j]==0:
				pass
			else:
				MI_TY+=TY_matrix[i,j]*np.log2(TY_matrix[i,j]/(P_layer_T_for_Y[i]*P_Y_for_layer_T[j]))

	return MI_XT,MI_TY

def Discretize_v2(layer_T):
	'''
	Discretize the output of the neuron
	Inputs:
	- layer_T:(N,H)
	Outputs:
	- layer_T:(N,H) the new layer_T after discretized
	'''

	NUM_INTERVALS = 30
	labels = np.arange(NUM_INTERVALS)
	bins = np.arange(NUM_INTERVALS+1)
	bins = bins/float(NUM_INTERVALS)

	for i in range(layer_T.shape[1]):
		temp = pd.cut(layer_T[:,i],bins,labels=labels)
		layer_T[:,i] = np.array(temp)
	return layer_T

