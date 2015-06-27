#from archconvnets.unsupervised.cudnn_module.cudnn_module import *
import time
import numpy as np
from scipy.io import savemat, loadmat
import copy
from scipy.stats import zscore
import random
import scipy

#global w_interp
n_in = 4
n_shifts = 3 # must be 3 [shift_w()]
n_controllers = 2
n_mem_slots = 5 # "M"
m_length = 8 # "N"

w_interp = np.random.random(size=(n_controllers, n_mem_slots))
gamma_weights = np.random.normal(size=(n_controllers, n_in))
interp_weights = np.random.random(size=(n_controllers, n_in))
shift_weights = np.random.normal(size=(n_controllers*n_shifts, n_in),scale=1)

w_prev = np.abs(np.random.normal(size=(n_controllers, n_mem_slots)))
w_content = np.abs(np.random.normal(size=(n_controllers, n_mem_slots)))

x = np.random.normal(size=(n_in,1))

t = np.random.normal(size=(n_controllers, n_mem_slots))


############# sharpen across mem_slots separately for each controller
def sharpen(layer_in, gamma_out):
	# layer_in: [n_controllers, n_mem_slots], gamma_out: [n_controllers, 1]
	layer_in_raised = layer_in**gamma_out
	return layer_in_raised / layer_in_raised.sum(1)[:,np.newaxis]

def sharpen_dlayer_in(layer_in, gamma_out, above_w=1):
	# layer_in, above_w: [n_controllers, n_mem_slots], gamma_out: [n_controllers, 1]
	layer_in_raised = layer_in**gamma_out
	layer_in_raised_m1 = layer_in**(gamma_out-1)
	denom = layer_in_raised.sum(1)[:,np.newaxis] # sum across slots
	denom2 = denom**2
	
	# dsharpen[:,i]/dlayer_in[:,j] when i = j:
	dsharpen_dlayer_in = above_w*(layer_in_raised_m1 * denom - layer_in_raised * layer_in_raised_m1)

	# dsharpen[:,i]/dlayer_in[:,j] when i != j:
	dsharpen_dlayer_in -=  np.einsum(layer_in_raised_m1, [0,1], above_w * layer_in_raised, [0,2], [0,1])
	dsharpen_dlayer_in += above_w * layer_in_raised_m1 * layer_in_raised
	
	dsharpen_dlayer_in *= gamma_out / denom2
	
	return dsharpen_dlayer_in

def sharpen_dgamma_out(layer_in, gamma_out, above_w=1):
	# layer_in, above_w: [n_controllers, n_mem_slots], gamma_out: [n_controllers, 1]
	w_gamma = layer_in**gamma_out
	w_gamma_sum = w_gamma.sum(1)[:,np.newaxis] # sum across slots
	
	ln_gamma = np.log(layer_in)
	
	dw_gamma_dgamma = w_gamma * ln_gamma
	dw_sum_gamma_dgamma = dw_gamma_dgamma.sum(1)[:,np.newaxis] # sum across slots
	
	dw_dgamma = (dw_gamma_dgamma * w_gamma_sum - w_gamma * dw_sum_gamma_dgamma) / (w_gamma_sum**2)
	return (dw_dgamma * above_w).sum(1)[:,np.newaxis]

#############
# relu below 'thresh' (0 = normal relu)
def relu(layer_in, thresh=0):
	temp = copy.deepcopy(layer_in)
	temp[layer_in < thresh] = thresh
	return temp

def relu_dlayer_in(layer_in, above_w=1, thresh=0):
	temp = above_w * np.ones_like(layer_in)
	temp[layer_in < thresh] = 0
	return temp
	

############
# softmax over second dimension; first dim. treated independently
def softmax(layer_in):
	exp_layer_in = np.exp(layer_in)
	return exp_layer_in/np.sum(exp_layer_in,1)[:,np.newaxis]

def softmax_dlayer_in(layer_out, above_w=1):
	# dsoftmax[:,i]/dlayer_in[:,j] when i = j:
	temp = above_w * (layer_out * (1 - layer_out))
	
	# dsoftmax[:,i]/dlayer_in[:,j] when i != j:
	temp += -np.einsum(np.squeeze(above_w*layer_out), [0,1], np.squeeze(layer_out), [0,2], [0,2]) + above_w*layer_out**2
	return temp

################# interpolate
def interpolate(w_content, w_prev, interp_gate_out):
	# w_prev, w_content: [n_controllers, n_mem_slots], interp_gate_out: [n_controllers, 1]
	return interp_gate_out * w_content + (1 - interp_gate_out) * w_prev

def interpolate_dinterp_gate_out(w_content, w_prev, above_w=1):
	return (above_w * (w_content - w_prev)).sum(1)[:,np.newaxis]

############### shift w
def shift_w(shift_out, w_interp):	
	# shift_out: [n_controllers, n_shifts], w_interp: [n_controllers, mem_length]
	
	w_tilde = np.zeros_like(w_interp)
	
	for loc in range(n_mem_slots):
		w_tilde[:,loc] = shift_out[:,0]*w_interp[:,loc-1] + shift_out[:,1]*w_interp[:,loc] + \
				shift_out[:,2]*w_interp[:,(loc+1)%n_mem_slots]
	return w_tilde # [n_controllers, mem_length]

def shift_w_dshift_out(w_interp, above_w=1):
	# w_interp: [n_controllers, mem_length], above_w: [n_controllers, mem_length]
	
	dshift_w_dshift_out = np.zeros((n_controllers, n_mem_slots, n_shifts))
	for m in range(n_mem_slots):
		for H in [-1,0,1]:
			dshift_w_dshift_out[:,m,H+1] = w_interp[:, (m+H)%n_mem_slots] * above_w[:,m]
	
	return dshift_w_dshift_out.sum(1) # [n_controllers, n_shifts]

def shift_w_dw_interp(shift_out, above_w=1):
	# shift_out: [n_controllers, n_shifts], above_w: [n_controllers, mem_length]
	
	temp = np.zeros_like(above_w)
	
	for loc in range(n_mem_slots):
		temp[:,loc-1] += above_w[:,loc] * shift_out[:,0]
		temp[:,loc] += above_w[:,loc] * shift_out[:,1]
		temp[:,(loc+1)%n_mem_slots] += above_w[:,loc] * shift_out[:,2]
			
	return temp # [n_controllers, mem_length]

############## linear layer
def linear_F(F, layer_in):
	# F: [n1, n_in], layer_in: [n_in, 1]
	
	return np.dot(F,layer_in) # [n1, 1]

def linear_dF(layer_in, above_w): 
	# layer_in: [n_in, 1], above_w: [n1,1]
	
	return layer_in.T*above_w # [n1, n_in]

def linear_dlayer_in(F, above_w=1):
	return (F*above_w).sum(0)[:,np.newaxis] # [n_in, 1]

################## squared layer
def sq_F(F, layer_in):
	# F: [n1, n_in], layer_in: [n_in, 1]
	
	return np.dot(F,layer_in)**2 # [n1, 1]

def sq_dF(F, layer_in, layer_out, above_w): 
	# F: [n1, n_in], layer_in: [n_in, 1], layer_out,above_w: [n1,1]
	
	s = np.sign(np.dot(F,layer_in))
	return 2*s*np.sqrt(layer_out)*(layer_in.T)*above_w # [n1, n_in]

def sq_dlayer_in(F, layer_in, layer_out, above_w=1):
	s = np.sign(np.dot(F,layer_in))
	return 2*(s*np.sqrt(layer_out)*F*above_w).sum(0)[:,np.newaxis] # [n_in, 1]
	
i_ind = 1
j_ind = 1


def f(y):
	shift_weights[i_ind,j_ind] = y
	#gamma_weights[i_ind,j_ind] = y
	#interp_weights[i_ind,j_ind] = y
	
	########
	# forward:
	
	# interpolation
	interp_gate_out = linear_F(interp_weights, x)
	w_interp = interpolate(w_content, w_prev, interp_gate_out)
	
	# shift
	shift_out = linear_F(shift_weights, x)
	shift_out_relu = relu(shift_out)
	
	shift_out_relu_smax = softmax(shift_out_relu.reshape((n_controllers, n_shifts)))
	w_tilde = shift_w(shift_out_relu_smax, w_interp)
	
	# sharpen
	gamma_out = linear_F(gamma_weights, x)
	gamma_out_relu = relu(gamma_out,1)
	
	w = sharpen(w_tilde, gamma_out_relu)
	
	return ((w - t)**2).sum()

def g(y):
	shift_weights[i_ind,j_ind] = y
	#gamma_weights[i_ind,j_ind] = y
	#interp_weights[i_ind,j_ind] = y
	
	########
	# forward:
	
	# interpolation
	interp_gate_out = linear_F(interp_weights, x)
	w_interp = interpolate(w_content, w_prev, interp_gate_out)
	
	# shift
	shift_out = linear_F(shift_weights, x)
	shift_out_relu = relu(shift_out)
	
	shift_out_relu_smax = softmax(shift_out_relu.reshape((n_controllers, n_shifts)))
	w_tilde = shift_w(shift_out_relu_smax, w_interp)
	
	# sharpen
	gamma_out = linear_F(gamma_weights, x)
	gamma_out_relu = relu(gamma_out,1)
	
	w = sharpen(w_tilde, gamma_out_relu)
	
	##########
	# backward (data):
	
	# sharpen
	dsharpen_dw_tilde = sharpen_dlayer_in(w_tilde, gamma_out_relu, 2*(w - t)) # sharpen
	dw_dgamma_out_relu = sharpen_dgamma_out(w_tilde, gamma_out_relu, 2*(w - t)) # sharpen
	dw_dgamma_out = relu_dlayer_in(gamma_out, dw_dgamma_out_relu, 1) # relu
	
	# shift
	dw_tilde_dshift_out = shift_w_dshift_out(w_interp, dsharpen_dw_tilde) # shift
	dshift_out_relu_smax_dshift_out = softmax_dlayer_in(shift_out_relu_smax, dw_tilde_dshift_out) # softmax
	dshift_out_relu_dshift_out = relu_dlayer_in(shift_out, dshift_out_relu_smax_dshift_out.ravel()[:,np.newaxis]) # relu

	# interpolation
	dw_tilde_dw_interp = shift_w_dw_interp(shift_out_relu_smax, dsharpen_dw_tilde) # shift
	dw_interp_dinterp_gate_out = interpolate_dinterp_gate_out(w_content, w_prev, dw_tilde_dw_interp) # interpolate
	
	
	##########
	# backward (vars):
	dgamma_out_dgamma_weights = linear_dF(x, dw_dgamma_out)
	dshift_out_dshift_weights = linear_dF(x, dshift_out_relu_dshift_out)
	dinterp_gate_out_dinterp_weights = linear_dF(x, dw_interp_dinterp_gate_out)
	
	#return dinterp_gate_out_dinterp_weights[i_ind,j_ind]
	#return dgamma_out_dgamma_weights[i_ind,j_ind]
	return dshift_out_dshift_weights[i_ind,j_ind]
	
np.random.seed(np.int64(time.time()))
eps = np.sqrt(np.finfo(np.float).eps)*1e1


N_SAMPLES = 25
ratios = np.zeros(N_SAMPLES)
for sample in range(N_SAMPLES):

	i_ind = np.random.randint(shift_weights.shape[0])
	j_ind = np.random.randint(shift_weights.shape[1])
	y = -1e0*shift_weights[i_ind,j_ind]; gt = g(y); gtx = scipy.optimize.approx_fprime(np.ones(1)*y, f, eps)
	
	'''i_ind = np.random.randint(interp_weights.shape[0])
	j_ind = np.random.randint(interp_weights.shape[1])
	y = -1e0*interp_weights[i_ind,j_ind]; gt = g(y); gtx = scipy.optimize.approx_fprime(np.ones(1)*y, f, eps)'''
	
	'''i_ind = np.random.randint(gamma_weights.shape[0])
	j_ind = np.random.randint(gamma_weights.shape[1])
	y = -1e0*gamma_weights[i_ind,j_ind]; gt = g(y); gtx = scipy.optimize.approx_fprime(np.ones(1)*y, f, eps)'''
	
	#i_ind = np.random.randint(gamma_weights.shape[0])
	#y = -1e0*gamma_weights[i_ind]; gt = g(y); gtx = scipy.optimize.approx_fprime(np.ones(1)*y, f, eps)
	
	
	if gtx == 0:
		ratios[sample] = 1
	else:
		ratios[sample] = gtx/gt
	print gt, gtx, ratios[sample]
	
print ratios.mean(), ratios.std()

