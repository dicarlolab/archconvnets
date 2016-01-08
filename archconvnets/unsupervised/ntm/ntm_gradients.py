import numpy as np
import copy
from init_vars import *
import archconvnets.unsupervised.ntm_module.ntm_module as nm

####

# pointwise multiply partials with scalar then add two sets of partials (A + B*s)
def pointwise_mult_partials_add__layers(A, B, s):
	assert len(A) == len(B)
	C = [None] * len(A)
	for layer in range(len(A)):
		C[layer] = A[layer] + B[layer] * s
	
	return C

def mult_partials(da_db, db_dc, b):
	#while da_db.ndim <= b.ndim:
	#	da_db = da_db[np.newaxis]
	
	a_ndim = da_db.ndim - b.ndim
	c_ndim = db_dc.ndim - b.ndim
	
	assert c_ndim > 0
	
	da_db_r = da_db.reshape((np.prod(da_db.shape[:a_ndim]), np.prod(da_db.shape[a_ndim:])))
	db_dc_r = db_dc.reshape((np.prod(db_dc.shape[:b.ndim]), np.prod(db_dc.shape[b.ndim:])))
	
	da_dc = np.dot(da_db_r, db_dc_r).reshape(np.concatenate((da_db.shape[:a_ndim], db_dc.shape[b.ndim:])))

	return da_dc

# pointwise multiply partials (same for all partials, i.e., broadcasted for dimensions taken wrt)
def pointwise_mult_partials__layers(mat, DB_DC):
	DB_DC_NEW = [None] * len(DB_DC)
	for layer in range(len(DB_DC)):
		# add singleton dimensions for broadcasting
		new_shape = mat.shape
		for add_dim in range(len(DB_DC[layer].shape) - len(mat.shape)):
			new_shape = np.append(new_shape, 1)
		
		mat_axes = mat.reshape(new_shape)
		
		DB_DC_NEW[layer] = mat_axes * DB_DC[layer]
	
	return DB_DC_NEW

# multiply list of partials
def mult_partials_chain(DA_DB, B_SZs):
	DA_DX = DA_DB[0]
	for x in range(1, len(DA_DB)):
		DA_DX = mult_partials(DA_DX, DA_DB[x], B_SZs[x-1])
	return DA_DX

# mult_partials for all layers in DB_DC (a list of matrices)
def mult_partials__layers(da_db, DB_DC, b, DB_DC_INIT=None):
	DC = [None] * len(DB_DC)
	for layer in range(len(DB_DC)):
		DC[layer] = mult_partials(da_db, DB_DC[layer], b)
		if DB_DC_INIT != None:
			DC[layer] += DB_DC_INIT[layer]
	
	return DC

##########
# focus keys, scalar beta_out (one for each controller) multiplied with each of its keys
def focus_keys(keys, beta_out):
	# keys: [n_controllers, m_length], beta_out: [n_controllers, 1]
	
	return keys * beta_out # [n_controllers, m_length]

def focus_key_dbeta_out(keys, beta_out): 
	# beta_out: [n_controllers, 1]
	n_controllers, m_length = keys.shape
	
	g = np.zeros((n_controllers, m_length, n_controllers, 1),dtype='single')
	g[range(n_controllers),:,range(n_controllers),0] = keys
	return g

def focus_key_dkeys(keys, beta_out): 
	# beta_out: [n_controllers, 1]
	n_controllers, m_length = keys.shape
	
	g = np.zeros((n_controllers, m_length, n_controllers, m_length),dtype='single')
	for j in range(m_length):
		g[range(n_controllers),j,range(n_controllers),j] = np.squeeze(beta_out)
	return g

############
# cosine similarity between each controller's key and memory vector

def cosine_sim_expand_dmem(keys, mem):
	n_controllers = keys.shape[0]
	comb = np.zeros((n_controllers, mem.shape[0], mem.shape[0], mem.shape[1]),dtype='single')

	keys_sq_sum = np.sqrt(np.sum(keys**2, 1))
	mem_sq_sum = np.sqrt(np.sum(mem**2, 1))

	denom = np.einsum(keys_sq_sum, [0], mem_sq_sum, [1], [0,1])
	numer = np.dot(keys, mem.T)

	numer = numer / denom**2
	denom = 1 / denom # = denom/denom**2

	mem = mem / mem_sq_sum[:,np.newaxis]

	temp = np.einsum(mem, [0,2], numer*keys_sq_sum[:,np.newaxis], [1,0], [1,0,2])
	
	keys_denom = keys[:,np.newaxis] * denom[:,:,np.newaxis]
	
	comb[:,range(mem.shape[0]),range(mem.shape[0])] = keys_denom - temp
	return comb

def cosine_sim_expand_dkeys(keys, mem):
	n_controllers = keys.shape[0]
	comb = np.zeros((n_controllers, mem.shape[0], n_controllers, keys.shape[1]),dtype='single')
	
	keys_sq_sum = np.sqrt(np.sum(keys**2, 1))
	mem_sq_sum = np.sqrt(np.sum(mem**2, 1))
	
	denom = np.einsum(keys_sq_sum, [0], mem_sq_sum, [1], [0,1])
	numer = np.dot(keys, mem.T)
	
	numer = numer / denom**2
	denom = 1 / denom # = denom/denom**2
	
	keys = keys / keys_sq_sum[:,np.newaxis]
	
	temp = np.einsum(keys, [1,2], numer*mem_sq_sum[np.newaxis], [1,0], [1,0,2])
	
	mem_denom = mem[np.newaxis] * denom[:,:,np.newaxis]
	
	comb[range(n_controllers),:,range(n_controllers)] = mem_denom - temp
	return comb

def cosine_sim(keys, mem):
	# keys [n_controllers, m_length], mem: [n_mem_slots, m_length]
	numer = np.dot(keys, mem.T)
	denom = np.einsum(np.sqrt(np.sum(keys**2,1)), [0], np.sqrt(np.sum(mem**2,1)), [1], [0,1])
	
	return numer / denom # [n_controllers, n_mem_slots]

############# sharpen across mem_slots separately for each controller
def sharpen(w, gamma):
	wg = w ** gamma
	return wg / wg.sum(1)[:,np.newaxis]

def dsharpen_dgamma(w, gamma):
	n = w.shape[0]
	g = np.zeros(np.concatenate((w.shape, gamma.shape)),dtype='single')
	
	wg = w ** gamma
	ln_w_wg = np.log(w)*wg
	wg_sum = wg.sum(1)[:,np.newaxis]
	ln_w_wg_sum = ln_w_wg.sum(1)[:,np.newaxis]
	
	t = (ln_w_wg * wg_sum - wg * ln_w_wg_sum) / (wg_sum ** 2)
	
	g[range(n),:,range(n)] = t[:,:,np.newaxis]
	
	return g
	
def dsharpen_dw(w, gamma):
	n = w.shape[0]
	g = np.zeros(np.concatenate((w.shape, w.shape)),dtype='single')
	
	wg = w ** gamma
	wg_sum = wg.sum(1)[:,np.newaxis]
	wg_sum2 = wg_sum ** 2
	g_wgm1 = gamma * (w ** (gamma-1))
	
	t = (g_wgm1 / wg_sum2) * (wg_sum - wg)
	
	for i in range(w.shape[0]):
		g[i,:,i,:] = t[i]
	
	for j in range(w.shape[1]):
		for b in range(w.shape[1]):
			if b != j:
				g[range(n),j,range(n),b] = -g_wgm1[:,b] * wg[:,j] / np.squeeze(wg_sum2)
	
	return g

##############
# sigmoid point-wise
def sigmoid(layer_in):
	return 1/(1+np.exp(-layer_in))

def sigmoid_dlayer_in(layer_out):
	d = layer_out * (1-layer_out)
	t = np.zeros(np.concatenate((layer_out.shape, layer_out.shape)),dtype='single')
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

def relu_dlayer_in(layer_in, thresh=0):
	temp = np.ones_like(layer_in)
	temp[layer_in <= thresh] = 0
	
	temp2 = np.zeros(np.concatenate((layer_in.shape, layer_in.shape)),dtype='single')
	for i in range(layer_in.shape[0]):
		for j in range(layer_in.shape[1]):
			temp2[i,j,i,j] = temp[i,j]
	return temp2
	

############
# softmax over second dimension; first dim. treated independently
def softmax(layer_in):
	exp_layer_in = np.exp(layer_in)
	return exp_layer_in/np.sum(exp_layer_in,1)[:,np.newaxis]

def softmax_dlayer_in(layer_out):
	g = np.zeros((layer_out.shape[0], layer_out.shape[1], layer_out.shape[0], layer_out.shape[1]),dtype='single')
	
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

def shift_w_dshift_out(w_interp):
	n_shifts = 3 #...
	
	temp = np.zeros((C, M, C, n_shifts),dtype='single')
	for m in range(M):
		for H in [-1,0,1]:
			temp[range(C),m,range(C),H+1] = w_interp[:, (m+H)%M]
	
	return temp

def shift_w_dw_interp(shift_out):
	# shift_out: [n_controllers, n_shifts]
	temp = np.zeros((C, M, C, M),dtype='single')
	
	for loc in range(M):
		temp[range(C),loc,range(C),loc-1] = shift_out[:,0]
		temp[range(C),loc,range(C),loc] = shift_out[:,1]
		temp[range(C),loc,range(C),(loc+1)%M] = shift_out[:,2]
			
	return temp

############## linear layer
def linear_F(F, layer_in):
	# F: [n1, n_in], layer_in: [n_in, 1]
	
	return np.dot(F,layer_in) # [n1, 1]

def linear_F_dx(F, x):
	n = x.shape[1]
	temp = np.zeros((F.shape[0], n, x.shape[0], n),dtype='single')
	temp[:,range(n),:,range(n)] = F
	return temp

def linear_F_dF(F, x):
	n = F.shape[0]
	temp = np.zeros((n, x.shape[1], n, F.shape[1]),dtype='single')
	temp[range(n),:,range(n)] = x.T
	return temp
	

############# softmax <- linear 2d
def linear_2d_F_softmax(ww,x):
	return softmax(linear_2d_F(ww,x))

def linear_2d_F_softmax_dF(out, ww,x):
	dout_dlin = softmax_dlayer_in(out)
	dlin_dF = linear_2d_F_dF(ww,x)
	return mult_partials(dout_dlin, dlin_dF, out)

def linear_2d_F_softmax_dx(out, ww):
	dout_dlin = softmax_dlayer_in(out)
	dlin_dF = linear_2d_F_dx(ww)
	return mult_partials(dout_dlin, dlin_dF, out)
	
#######
def linear_2d_F(ww,x):
	return np.squeeze(np.dot(ww,x))

def linear_2d_F_dF(ww,x):
	n = ww.shape[1]
	temp = np.zeros((ww.shape[0], n, ww.shape[0], n, ww.shape[2]),dtype='single')
	for i in range(ww.shape[0]):
		temp[i,range(n),i,range(n)] += np.squeeze(x)
	return temp

def linear_2d_F_dx(ww):
	return ww

##################
def sq_points(input):
	return input**2

def sq_points_dinput(input):
	n = input.shape[1]
	dinput = np.zeros((input.shape[0], n, input.shape[0], n),dtype='single')
	for i in range(input.shape[0]):
		dinput[i,range(n),i,range(n)] = 2*input[i]
	return dinput

#####
def add_mem(gw, add_out):
	return np.dot(gw.T, add_out)

def add_mem_dadd_out(gw):
	temp = np.zeros((M, mem_length, C, mem_length),dtype='single')
	temp[:,range(mem_length),:,range(mem_length)] = gw.T
	return temp

def add_mem_dgw(add_out):
	temp = np.zeros((M, mem_length, C, M),dtype='single')
	temp[range(M),:,:,range(M)] = add_out.T
	return temp


################# softmax <- interpolate
def interpolate_softmax(interp_gate_out, o_content, o_prev):
	return softmax(interpolate(interp_gate_out, o_content, o_prev))

def interpolate_softmax_dinterp_gate_out(out, interp_gate_out, o_content, o_prev):
	dout_dlin = nm.softmax_dlayer_in_cpu(out)
	dlin_dinterp_gate_out = interpolate_dinterp_gate_out(interp_gate_out, o_content, o_prev)
	return mult_partials(dout_dlin, dlin_dinterp_gate_out, out)
	

def interpolate_softmax_do_content(out, interp_gate_out, o_content):
	dout_dlin = nm.softmax_dlayer_in_cpu(out)
	dlin_do_content = interpolate_do_content(interp_gate_out, o_content)
	return mult_partials(dout_dlin, dlin_do_content, out)

def interpolate_softmax_do_prev(out, o_gatei, o_previ):
	dout_dlin = nm.softmax_dlayer_in_cpu(out)
	dlin_do_prev = interpolate_do_prev(o_gatei, o_previ)
	return mult_partials(dout_dlin, dlin_do_prev, out)

	
################# interpolate
def interpolate(interp_gate_out, o_content, o_prev):
	print interp_gate_out.shape, o_content.shape, o_prev.shape
	return interp_gate_out * o_content + (1 - interp_gate_out) * o_prev

def interpolate_dinterp_gate_out(interp_gate_out, o_content, o_prev):
	temp = o_content - o_prev
	temp2 = np.zeros((temp.shape[0], temp.shape[1], interp_gate_out.shape[0], 1),dtype='single')
	
	for i in range(temp2.shape[0]):
		for j in range(temp2.shape[1]):
			temp2[i,j,i] = temp[i,j]
	return temp2

def interpolate_do_content(interp_gate_out, o_content):
	temp = interp_gate_out
	n = o_content.shape[1]
	temp2 = np.zeros((o_content.shape[0], n, o_content.shape[0], n),dtype='single')
	
	for i in range(temp2.shape[0]):
		temp2[i,range(n),i,range(n)] = temp[i]
	return temp2

def interpolate_do_prev(interp_gate_out, o_previ):
	temp = 1 - interp_gate_out
	n = o_previ.shape[1]
	temp2 = np.zeros((o_previ.shape[0], n, o_previ.shape[0], n),dtype='single')
	
	for i in range(temp2.shape[0]):
		temp2[i,range(n),i,range(n)] = temp[i]
	return temp2
