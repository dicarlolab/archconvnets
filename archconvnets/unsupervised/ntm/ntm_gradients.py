import numpy as np
import copy
from init_vars import *

####
def mult_partials(da_db, db_dc, b):
	a_ndim = da_db.ndim - b.ndim
	c_ndim = db_dc.ndim - b.ndim
	keep_dims = np.concatenate((range(a_ndim), range(da_db.ndim, da_db.ndim + c_ndim)))
	da_dc = np.einsum(da_db, range(da_db.ndim), db_dc, range(a_ndim, a_ndim + db_dc.ndim), keep_dims)
	return da_dc

# collapse (sum) over output dims
def mult_partials_collapse(da_db, db_dc, b):
	a_ndim = da_db.ndim - b.ndim
	da_dc = mult_partials(da_db, db_dc, b)
	dc = np.einsum(da_dc, range(da_dc.ndim), range(a_ndim, da_dc.ndim))
	return dc

# mult_partials_collapse for all layers in DB_DC (a list of matrices)
def mult_partials_collapse__layers(da_db, DB_DC, b, DB_DC_INIT=None):
	DC = [None] * len(DB_DC)
	for layer in range(len(DB_DC)):
		DC[layer] = mult_partials_collapse(da_db, DB_DC[layer], b)
		if DB_DC_INIT != None:
			DC[layer] += DB_DC_INIT[layer]
	
	return DC

# mult_partials for all layers in DB_DC (a list of matrices)
def mult_partials__layers(da_db, DB_DC, b, DB_DC_INIT=None):
	DC = [None] * len(DB_DC)
	for layer in range(len(DB_DC)):
		DC[layer] = mult_partials(da_db, DB_DC[layer], b)
		if DB_DC_INIT != None:
			DC[layer] += DB_DC_INIT[layer]
	
	return DC

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

def focus_key_dbeta_out_nsum(keys, beta_out): 
	# beta_out: [n_controllers, 1]
	n_controllers, m_length = keys.shape
	
	g = np.zeros((n_controllers, m_length, n_controllers, 1))
	g[range(n_controllers),:,range(n_controllers),0] = keys
	return g

def focus_key_dkeys_nsum(keys, beta_out): 
	# beta_out: [n_controllers, 1]
	n_controllers, m_length = keys.shape
	
	g = np.zeros((n_controllers, m_length, n_controllers, m_length))
	for j in range(m_length):
		g[range(n_controllers),j,range(n_controllers),j] = np.squeeze(beta_out)
	return g

############
# cosine similarity between each controller's key and memory vector

def cosine_sim_expand_dmem(keys, mem):
	n_controllers = keys.shape[0]
	dnumer = np.zeros((n_controllers, mem.shape[0], mem.shape[0], mem.shape[1]))
	ddenom = np.zeros_like(dnumer); comb = np.zeros_like(dnumer)
	
	for j in range(mem.shape[0]):
		dnumer[:,j,j] = keys
	
	denom = np.einsum(np.sqrt(np.sum(keys**2,1)), [0], np.sqrt(np.sum(mem**2,1)), [1], [0,1])
	numer = np.dot(keys, mem.T)
	
	for i in range(keys.shape[0]):
		for j in range(mem.shape[0]):
			ddenom[i,j,j] = mem[j] * np.sqrt(np.sum(keys[i]**2)) / np.sqrt(np.sum(mem[j]**2))
			comb[i,j,j] = (dnumer[i,j,j] * denom[i,j] - numer[i,j] * ddenom[i,j,j])/(denom[i,j]**2)
	return comb

def cosine_sim_expand_dkeys(keys, mem):
	n_controllers = keys.shape[0]
	dnumer = np.zeros((n_controllers, mem.shape[0], n_controllers, keys.shape[1]))
	ddenom = np.zeros_like(dnumer); comb = np.zeros_like(dnumer)
	
	dnumer[range(n_controllers),:,range(n_controllers)] = mem
	
	denom = np.einsum(np.sqrt(np.sum(keys**2,1)), [0], np.sqrt(np.sum(mem**2,1)), [1], [0,1])
	numer = np.dot(keys, mem.T)
	
	for i in range(keys.shape[0]):
		for j in range(mem.shape[0]):
			ddenom[i,j,i] = keys[i] * np.sqrt(np.sum(mem[j]**2)) / np.sqrt(np.sum(keys[i]**2))
			comb[i,j,i] = (dnumer[i,j,i] * denom[i,j] - numer[i,j] * ddenom[i,j,i])/(denom[i,j]**2)
	return comb

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

def sigmoid_dlayer_in(layer_out):
	d = layer_out * (1-layer_out)
	t = np.zeros(np.concatenate((layer_out.shape, layer_out.shape)))
	for i in range(layer_out.shape[0]):
		for j in range(layer_out.shape[1]):
			t[i,j,i,j] = d[i,j]
	return t


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

def softmax_dlayer_in_nsum(layer_out):
	g = np.zeros((layer_out.shape[0], layer_out.shape[1], layer_out.shape[0], layer_out.shape[1]))
	
	# dsoftmax[:,i]/dlayer_in[:,j] when i = j:
	temp = (layer_out * (1 - layer_out))
	for i in range(g.shape[0]):
		for j in range(g.shape[1]):
			g[i,j,i,j] = temp[i,j]
	
	# i != j
	for i in range(g.shape[0]):
		for j in range(g.shape[1]):
			for k in range(g.shape[1]):
				if j != k:
					g[i,j,i,k] -= layer_out[i,j]*layer_out[i,k]
	return g

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

def shift_w_dshift_out_nsum(w_interp):
	n_shifts = 3 #...
	
	temp = np.zeros((C, M, C, n_shifts))
	for m in range(M):
		for H in [-1,0,1]:
			temp[range(C),m,range(C),H+1] = w_interp[:, (m+H)%M]
	
	return temp

def shift_w_dw_interp(shift_out, above_w=1):
	# shift_out: [n_controllers, n_shifts], above_w: [n_controllers, mem_length]
	
	temp = np.zeros_like(above_w)
	n_mem_slots = above_w.shape[1]
	
	for loc in range(n_mem_slots):
		temp[:,loc-1] += above_w[:,loc] * shift_out[:,0]
		temp[:,loc] += above_w[:,loc] * shift_out[:,1]
		temp[:,(loc+1)%n_mem_slots] += above_w[:,loc] * shift_out[:,2]
			
	return temp # [n_controllers, mem_length]

def shift_w_dw_interp_nsum(shift_out):
	# shift_out: [n_controllers, n_shifts]
	temp = np.zeros((C, M, C, M))
	
	for loc in range(M):
		temp[range(C),loc,range(C),loc-1] = shift_out[:,0]
		temp[range(C),loc,range(C),loc] = shift_out[:,1]
		temp[range(C),loc,range(C),(loc+1)%M] = shift_out[:,2]
			
	return temp

############# linear then sigmoid
def linear_F_sigmoid(F, layer_in):
	# F: [n1, n_in], layer_in: [n_in, 1]
	
	return sigmoid(linear_F(F, layer_in)) # [n1, 1]

def linear_F_sigmoid_dF_nsum_g(out, F, mem):
	dout_dlin = sigmoid_dlayer_in(out)
	
	dlin_dF = linear_F_dF_nsum_g(F, mem)
	
	return mult_partials(dout_dlin, dlin_dF, out)
	
def linear_F_sigmoid_dx_nsum_g(out, F, mem):
	dout_dlin = sigmoid_dlayer_in(out)
	
	dlin_dx = linear_F_dx_nsum_g(F, mem)
	
	return mult_partials(dout_dlin, dlin_dx, out)
	
############## linear layer
def linear_F(F, layer_in):
	# F: [n1, n_in], layer_in: [n_in, 1]
	
	return np.dot(F,layer_in) # [n1, 1]

def linear_dF(layer_in, above_w): 
	# layer_in: [n_in, 1], above_w: [n1,1]
	
	return layer_in.T*above_w # [n1, n_in]

def linear_dlayer_in(F, above_w=1):
	return (F*above_w).sum(0)[:,np.newaxis] # [n_in, 1]

def linear_F_dx_nsum(o):
	n = mem_previ.shape[1]
	temp = np.zeros((OR_PREVi[F].shape[0], n, mem_previ.shape[0], n))
	temp[:,range(n),:,range(n)] = o
	return temp

def linear_F_dx_nsum_g(o, mem):
	n = mem.shape[1]
	temp = np.zeros((o.shape[0], n, mem.shape[0], n))
	temp[:,range(n),:,range(n)] = o
	return temp

def linear_F_dF_nsum(mem):
	n = OR_PREVi[F].shape[0]
	temp = np.zeros((n, mem.shape[1], n, OR_PREVi[F].shape[1]))
	temp[range(n),:,range(n)] = mem.T
	return temp

def linear_F_dF_nsum_g(F, mem):
	n = F.shape[0]
	temp = np.zeros((n, mem.shape[1], n, F.shape[1]))
	temp[range(n),:,range(n)] = mem.T
	return temp
	
################## squared layer
def sq_F(F, layer_in):
	# F: [n1, n_in], layer_in: [n_in, 1]
	
	return np.dot(F,layer_in)**2 # [n1, 1]

def sq_dF_nsum(F, layer_in, layer_out): 
	# F: [n1, n_in], layer_in: [n_in, 1], layer_out,above_w: [n1,1]
	
	s = np.sign(np.dot(F,layer_in))
	temp = 2*s*np.sqrt(layer_out)*(layer_in.T) # [n1, n_in]
	
	temp2 = np.zeros((temp.shape[0], temp.shape[0], temp.shape[1]))
	for i in range(temp.shape[0]):
		temp2[i,i] = temp[i]
	return temp2

def sq_dF(F, layer_in, layer_out, above_w=1): 
	# F: [n1, n_in], layer_in: [n_in, 1], layer_out,above_w: [n1,1]
	
	s = np.sign(np.dot(F,layer_in))
	return 2*s*np.sqrt(layer_out)*(layer_in.T)*above_w # [n1, n_in]

def sq_dlayer_in(F, layer_in, layer_out, above_w=1):
	s = np.sign(np.dot(F,layer_in))
	return 2*(s*np.sqrt(layer_out)*F*above_w).sum(0)[:,np.newaxis] # [n_in, 1]

def sq_dlayer_in_nsum(F, layer_in): # not summed across n1 as sq_dlayer_in() does
	return 2 * F * linear_F(F,layer_in) # [n1, n_in]

############# softmax <- linear 2d
def linear_2d_F_softmax(ww,x):
	return softmax(linear_2d_F(ww,x))

def linear_2d_F_softmax_dF_nsum(out, ww,x):
	dout_dlin = softmax_dlayer_in_nsum(out)
	dlin_dF = linear_2d_F_dF_nsum(ww,x)
	return mult_partials(dout_dlin, dlin_dF, out)

def linear_2d_F_softmax_dx_nsum(out, ww):
	dout_dlin = softmax_dlayer_in_nsum(out)
	dlin_dF = linear_2d_F_dx_nsum(ww)
	return mult_partials(dout_dlin, dlin_dF, out)
	
#######
def linear_2d_F(ww,x):
	return np.squeeze(np.dot(ww,x))

def linear_2d_F_dF_nsum(ww,x):
	n = ww.shape[1]
	temp = np.zeros((ww.shape[0], n, ww.shape[0], n, ww.shape[2]))
	for i in range(ww.shape[0]):
		temp[i,range(n),i,range(n)] += np.squeeze(x)
	return temp

def linear_2d_F_dx_nsum(ww):
	return ww

##################
def sq_points(input):
	return input**2

def sq_points_dinput(input):
	n = input.shape[1]
	dinput = np.zeros((input.shape[0], n, input.shape[0], n))
	for i in range(input.shape[0]):
		dinput[i,range(n),i,range(n)] = 2*input[i]
	return dinput

#####
def add_mem(gw, add_out):
	return np.dot(gw.T, add_out)

def add_mem_dadd_out(gw):
	temp = np.zeros((M, mem_length, C, mem_length))
	temp[:,range(mem_length),:,range(mem_length)] = gw.T
	return temp

def add_mem_dgw(add_out):
	temp = np.zeros((M, mem_length, C, M))
	temp[range(M),:,:,range(M)] = add_out.T
	return temp


################# softmax <- interpolate
def interpolate_softmax(interp_gate_out, o_content, o_prev):
	return softmax(interpolate(interp_gate_out, o_content, o_prev))

def interpolate_softmax_dinterp_gate_out(out, interp_gate_out, o_content, o_prev):
	dout_dlin = softmax_dlayer_in_nsum(out)
	dlin_dinterp_gate_out = interpolate_dinterp_gate_out(interp_gate_out, o_content, o_prev)
	return mult_partials(dout_dlin, dlin_dinterp_gate_out, out)
	

def interpolate_softmax_do_content(out, interp_gate_out, o_content):
	dout_dlin = softmax_dlayer_in_nsum(out)
	dlin_do_content = interpolate_do_content(interp_gate_out, o_content)
	return mult_partials(dout_dlin, dlin_do_content, out)

def interpolate_softmax_do_prev(out, o_gatei, o_previ):
	dout_dlin = softmax_dlayer_in_nsum(out)
	dlin_do_prev = interpolate_do_prev(o_gatei, o_previ)
	return mult_partials(dout_dlin, dlin_do_prev, out)

	
################# interpolate
def interpolate(interp_gate_out, o_content, o_prev):
	return interp_gate_out * o_content + (1 - interp_gate_out) * o_prev

def interpolate_dinterp_gate_out(interp_gate_out, o_content, o_prev):
	temp = o_content - o_prev
	temp2 = np.zeros((temp.shape[0], temp.shape[1], interp_gate_out.shape[0], 1))
	
	for i in range(temp2.shape[0]):
		for j in range(temp2.shape[1]):
			temp2[i,j,i] = temp[i,j]
	return temp2

def interpolate_do_content(interp_gate_out, o_content):
	temp = interp_gate_out
	n = o_content.shape[1]
	temp2 = np.zeros((o_content.shape[0], n, o_content.shape[0], n))
	
	for i in range(temp2.shape[0]):
		temp2[i,range(n),i,range(n)] = temp[i]
	return temp2

def interpolate_do_prev(o_gatei, o_previ):
	temp = 1 - o_gatei
	n = o_previ.shape[1]
	temp2 = np.zeros((o_previ.shape[0], n, o_previ.shape[0], n))
	
	for i in range(temp2.shape[0]):
		temp2[i,range(n),i,range(n)] = temp[i]
	return temp2
