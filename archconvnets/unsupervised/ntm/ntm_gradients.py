import numpy as np
import copy

##### read memory vector (generalization of linear_F())
def read_from_mem(w, mem):
	# w: [n_controllers, n_mem_slots], mem: [n_mem_slots, m_length]
	return np.dot(w, mem) # [n_controllers, m_length]

def read_from_mem_dw(mem, above_w):
	return np.dot(above_w, mem.T)

def read_from_mem_dmem(w, above_w):
	return np.dot(w.T, above_w)

##########
# focus keys, scalar beta_out (one for each controller) multiplied with each of its keys
def focus_keys(keys, beta_out):
	# keys: [n_controllers, m_length], beta_out: [n_controllers, 1]
	
	return keys * beta_out # [n_controllers, m_length]

def focus_key_dkeys(beta_out, above_w): 
	# above_w: [n_controllers, m_length], beta_out: [n_controllers, 1]
	
	return above_w * beta_out # [n_controllers, m_length]

def focus_key_dbeta_out(keys, above_w): 
	# above_w: [n_controllers, m_length], beta_out: [n_controllers, 1]
	
	return (above_w * keys).sum(1) # [n_controllers, 1]

############
# cosine similarity between each controller's key and memory vector
def cosine_sim(keys, mem):
	# keys [n_controllers, m_length], mem: [n_mem_slots, m_length]
	numer = np.dot(keys, mem.T)
	denom = np.einsum(np.sqrt(np.sum(keys**2,1)), [0], np.sqrt(np.sum(mem**2,1)), [1], [0,1])
	
	return numer / denom # [n_controllers, n_mem_slots]

def cosine_sim_dkeys(keys, mem, above_w=1):
	# keys [n_controllers, m_length], mem: [n_mem_slots, m_length]
	numer = np.dot(keys, mem.T)
	denom = np.einsum(np.sqrt(np.sum(keys**2,1)), [0], np.sqrt(np.sum(mem**2,1)), [1], [0,1])
	above_w_denom2 = above_w/(denom**2)
	
	denom_keys = np.sqrt(np.sum(keys**2,1))
	denom_mem = np.sqrt(np.sum(mem**2,1))
	
	dnumer_keys = np.dot(denom*above_w_denom2, mem)
	ddenom_keys = keys * (np.dot(numer*above_w_denom2, denom_mem)/denom_keys)[:,np.newaxis]

	return dnumer_keys - ddenom_keys # [n_controllers, m_length]
	
def cosine_sim_dmem(keys, mem, above_w=1):
	# keys [n_controllers, m_length], mem: [n_mem_slots, m_length]
	numer = np.dot(keys, mem.T)
	denom = np.einsum(np.sqrt(np.sum(keys**2,1)), [0], np.sqrt(np.sum(mem**2,1)), [1], [0,1])
	above_w_denom2 = above_w/(denom**2)
	
	denom_keys = np.sqrt(np.sum(keys**2,1))
	denom_mem = np.sqrt(np.sum(mem**2,1))
	
	dnumer_mem = np.dot((denom*above_w_denom2).T, keys)
	ddenom_mem = mem * (np.dot((numer*above_w_denom2).T, denom_keys)/denom_mem)[:,np.newaxis]

	return dnumer_mem - ddenom_mem # [n_controllers, m_length]

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

##############
# sigmoid point-wise
def sigmoid(layer_in):
	return 1/(1+np.exp(-layer_in))

def sigmoid_dlayer_in(layer_out, above_w):
	return above_w * layer_out * (1-layer_out)


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
	return (above_w * (w_content - w_prev)).sum(1)[:,np.newaxis] # sum across mem_slots

def interpolate_dw_content(interp_gate_out, above_w=1):
	return above_w * interp_gate_out

# todo: interpolate_dw_prev()

############### shift w
def shift_w(shift_out, w_interp):	
	# shift_out: [n_controllers, n_shifts], w_interp: [n_controllers, mem_length]
	
	w_tilde = np.zeros_like(w_interp)
	n_mem_slots = w_interp.shape[1]
	
	for loc in range(n_mem_slots):
		w_tilde[:,loc] = shift_out[:,0]*w_interp[:,loc-1] + shift_out[:,1]*w_interp[:,loc] + \
				shift_out[:,2]*w_interp[:,(loc+1)%n_mem_slots]
	return w_tilde # [n_controllers, mem_length]

def shift_w_dshift_out(w_interp, above_w=1):
	# w_interp: [n_controllers, mem_length], above_w: [n_controllers, mem_length]
	
	n_controllers = w_interp.shape[0]
	n_mem_slots = above_w.shape[1]
	n_shifts = 3 #...
	
	dshift_w_dshift_out = np.zeros((n_controllers, n_mem_slots, n_shifts))
	for m in range(n_mem_slots):
		for H in [-1,0,1]:
			dshift_w_dshift_out[:,m,H+1] = w_interp[:, (m+H)%n_mem_slots] * above_w[:,m]
	
	return dshift_w_dshift_out.sum(1) # [n_controllers, n_shifts]

def shift_w_dw_interp(shift_out, above_w=1):
	# shift_out: [n_controllers, n_shifts], above_w: [n_controllers, mem_length]
	
	temp = np.zeros_like(above_w)
	n_mem_slots = above_w.shape[1]
	
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
	