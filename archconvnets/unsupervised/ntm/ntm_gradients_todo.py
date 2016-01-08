
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


