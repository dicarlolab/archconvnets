
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


