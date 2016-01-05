
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


################# interpolate
def interpolate(interp_gate_out, o_content, o_prev):
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
