from archconvnets.unsupervised.cudnn_module.cudnn_module import *
import time
import numpy as np
from scipy.io import savemat, loadmat
import copy
from scipy.stats import zscore
import random
import scipy

n_in = 2
n1 = 3
n2 = 4
n3 = 6

F1 = np.random.random(size=(n1, n_in))
F2 = np.random.random(size=(n2, n1))
F3 = np.random.random(size=(n3, n2))

x = np.random.random(size=(n_in,1))

def t1(F, layer_in):
	return np.dot(F,layer_in)**2 # [n1, 1]

def t1_dF(F, layer_in, layer_out, tt1_dt1):
	s = np.sign(np.dot(F,layer_in))
	return 2*s*np.sqrt(layer_out)*(layer_in.T)*tt1_dt1 # [n1, n_in]

def t1_dx(F, layer_in, layer_out, tt1_dt1=1):
	s = np.sign(np.dot(F,layer_in))
	return 2*(s*np.sqrt(layer_out)*F*tt1_dt1).sum(0)[:,np.newaxis] # [n_in, 1]
	
i_ind = 1
j_ind = 1

def f(y):
	#F1[i_ind,j_ind] = y
	F2[i_ind,j_ind] = y
	#x[i_ind] = y
	
	y1 = t1(F1,x)
	y2 = t1(F2, y1)
	y3 = t1(F3, y2)
	
	return y3.sum()

def g(y):
	#F1[i_ind,j_ind] = y
	F2[i_ind,j_ind] = y
	#x[i_ind] = y
	
	y1 = t1(F1, x)
	y2 = t1(F2, y1)
	y3 = t1(F3, y2)
	
	dy3_dy2 = t1_dx(F3, y2, y3)
	dy2_dy1 = t1_dx(F2, y1, y2, dy3_dy2)
	
	dy2_dF = t1_dF(F2, y1, y2, dy3_dy2)
	dy1_dF = t1_dF(F1, x, y1, dy2_dy1)
	
	return dy2_dF[i_ind,j_ind]
	
np.random.seed(np.int64(time.time()))
eps = np.sqrt(np.finfo(np.float).eps)*1e3#9#10#10


N_SAMPLES = 25
ratios = np.zeros(N_SAMPLES)
for sample in range(N_SAMPLES):

	i_ind = np.random.randint(F2.shape[0])
	j_ind = np.random.randint(F2.shape[1])
	y = -1e0*F2[i_ind,j_ind]; gt = g(y); gtx = scipy.optimize.approx_fprime(np.ones(1)*y, f, eps);

	#i_ind = np.random.randint(F1.shape[0])
	#j_ind = np.random.randint(F1.shape[1])
	#y = -1e0*F1[i_ind,j_ind]; gt = g(y); gtx = scipy.optimize.approx_fprime(np.ones(1)*y, f, eps);
	
	#i_ind = np.random.randint(x.shape[0])
	#y = -1e0*x[i_ind]; gt = g(y); gtx = scipy.optimize.approx_fprime(np.ones(1)*y, f, eps);
	
	
	if gtx == 0:
		ratios[sample] = 1
	else:
		ratios[sample] = gtx/gt
	print gt, gtx, ratios[sample]
	
print ratios.mean(), ratios.std()

