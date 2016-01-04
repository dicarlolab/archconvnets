
############
##### check!
# softmax over second dimension; first dim. treated independently
def softmax(args):
	assert len(args) == 1
	layer_in = args[0]
	exp_layer_in = np.exp(layer_in)
	return exp_layer_in/np.sum(exp_layer_in,1)[:,np.newaxis]

def softmax_dlayer_in(args, layer_out):
	assert len(args) == 1
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

##########
# focus keys, scalar beta_out (one for each controller) multiplied with each of its keys
def focus_keys(args):
	keys, beta_out = args
	# keys: [n_controllers, m_length], beta_out: [n_controllers, 1]
	
	return keys * beta_out # [n_controllers, m_length]

def focus_key_dbeta_out(args, layer_out): 
	keys, beta_out = args
	# beta_out: [n_controllers, 1]
	n_controllers, m_length = keys.shape
	
	g = np.zeros((n_controllers, m_length, n_controllers, 1),dtype='single')
	g[range(n_controllers),:,range(n_controllers),0] = keys
	return g

def focus_key_dkeys(args, layer_out): 
	keys, beta_out = args
	# beta_out: [n_controllers, 1]
	n_controllers, m_length = keys.shape
	
	g = np.zeros((n_controllers, m_length, n_controllers, m_length),dtype='single')
	for j in range(m_length):
		g[range(n_controllers),j,range(n_controllers),j] = np.squeeze(beta_out)
	return g
