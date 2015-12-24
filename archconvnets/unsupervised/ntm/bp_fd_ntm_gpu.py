import time
import numpy as np
from scipy.io import savemat, loadmat
import copy
from scipy.stats import zscore
import random
import scipy
from ntm_gradients import *
from init_vars import *
from ntm_core_gpu import *
import archconvnets.unsupervised.ntm_module.ntm_module as nm

##### which gradients to test
#DERIV_L = L1_UNDER
#DERIV_L = L2_UNDER
#DERIV_L = F_UNDER

DERIV_L = L1_ABOVE
#DERIV_L = F_ABOVE

#DERIV_L = SHIFT
#DERIV_L = IN_GATE
#DERIV_L = KEY
#DERIV_L = BETA
#DERIV_L = ADD
#DERIV_L = ERASE
#DERIV_L = GAMMA

#gradient_category = 'write'
#gradient_category = 'read'
#gradient_category = 'under'
gradient_category = 'above'

#gradient_weights = False # false means bias terms
gradient_weights = True

def set_list_buffer(ind_counter, DATA):
	IND_LIST = [None]*len(DATA)
	SHAPE_LIST = [None]*len(DATA)
	
	for i in range(len(DATA)):
		if DATA[i] is not None:
			IND_LIST[i] = ind_counter; ind_counter += 1
			SHAPE_LIST[i] = DATA[i].shape
			nm.set_buffer(DATA[i], IND_LIST[i])
	return ind_counter, IND_LIST, SHAPE_LIST

####
if gradient_category == 'above':
	ref = WABOVEi[DERIV_L]
elif gradient_category == 'under':
	if gradient_weights:
		ref = WUNDERi[DERIV_L]
	else:
		ref = BUNDERi[DERIV_L]
elif gradient_category == 'read':
	if gradient_weights:
		ref = WRi[DERIV_L]
	else:
		ref = BRi[DERIV_L]
else:
	if gradient_weights:
		ref = WWi[DERIV_L]
	else:
		ref = BWi[DERIV_L]



########
def f(y):
	WW = copy.deepcopy(WWi); WR = copy.deepcopy(WRi);
	BW = copy.deepcopy(BWi); BR = copy.deepcopy(BRi);
	WUNDER = copy.deepcopy(WUNDERi); BUNDER = copy.deepcopy(BUNDERi);
	WABOVE = copy.deepcopy(WABOVEi); BABOVE = copy.deepcopy(BABOVEi);
	##
	
	if gradient_category == 'above':
		WABOVE[DERIV_L][i_ind,j_ind] = y
	elif ref.ndim == 2 and gradient_category == 'under':
		if gradient_weights:
			WUNDER[DERIV_L][i_ind,j_ind] = y
		else:
			BUNDER[DERIV_L][i_ind,j_ind] = y
	elif ref.ndim == 2 and gradient_category == 'read':
		if gradient_weights:
			WR[DERIV_L][i_ind,j_ind] = y
		else:
			BR[DERIV_L][i_ind,j_ind] = y
	elif gradient_category == 'read':
		if gradient_weights:
			WR[DERIV_L][i_ind,j_ind,k_ind] = y
		else:
			BR[DERIV_L][i_ind,j_ind,k_ind] = y
	elif ref.ndim == 2:
		if gradient_weights:
			WW[DERIV_L][i_ind,j_ind] = y
		else:
			BW[DERIV_L][i_ind,j_ind] = y
	else:
		if gradient_weights:
			WW[DERIV_L][i_ind,j_ind,k_ind] = y
		else:
			BW[DERIV_L][i_ind,j_ind,k_ind] = y
	##
	
	OR_PREV = copy.deepcopy(OR_PREVi); OW_PREV = copy.deepcopy(OW_PREVi)
	mem_prev = copy.deepcopy(mem_previ)
	
	for frame in range(1,N_FRAMES+1):
		OR_PREV,OW_PREV,mem_prev,read_mem,junk,OABOVE = forward_pass(WUNDER, BUNDER, WR,WW,BR,BW, WABOVE, BABOVE,OR_PREV, OW_PREV, mem_prev, x[frame])
		
	return ((OABOVE[F_ABOVE] - t)**2).sum()


def g(y):
	WW = copy.deepcopy(WWi); WR = copy.deepcopy(WRi);
	BW = copy.deepcopy(BWi); BR = copy.deepcopy(BRi);
	WUNDER = copy.deepcopy(WUNDERi); BUNDER = copy.deepcopy(BUNDERi);
	WABOVE = copy.deepcopy(WABOVEi); BABOVE = copy.deepcopy(BABOVEi);
	OR_PREV = copy.deepcopy(OR_PREVi); OW_PREV = copy.deepcopy(OW_PREVi)
	OW_PREV_PREV = copy.deepcopy(OW_PREV_PREVi); OUNDER_PREV = copy.deepcopy(OUNDER_PREVi)
	DOR_DWR = copy.deepcopy(DOR_DWRi); DOW_DWW = copy.deepcopy(DOW_DWWi); DOR_DWW = copy.deepcopy(DOR_DWWi)
	DOR_DBR = copy.deepcopy(DOR_DBRi); DOW_DBW = copy.deepcopy(DOW_DBWi); DOR_DBW = copy.deepcopy(DOR_DBWi)
	DOW_DWUNDER = copy.deepcopy(DOW_DWUNDERi); DOR_DWUNDER = copy.deepcopy(DOR_DWUNDERi)
	DOW_DBUNDER = copy.deepcopy(DOW_DBUNDERi); DOR_DBUNDER = copy.deepcopy(DOR_DBUNDERi)
	mem_prev = copy.deepcopy(mem_previ); mem_prev_prev = copy.deepcopy(mem_previ)
	DMEM_PREV_DWW = copy.deepcopy(DMEM_PREV_DWWi); DMEM_PREV_DBW = copy.deepcopy(DMEM_PREV_DBWi)
	DMEM_PREV_DWUNDER = copy.deepcopy(DMEM_PREV_DWUNDERi); DMEM_PREV_DBUNDER = copy.deepcopy(DMEM_PREV_DBUNDERi)
	
	##
	if gradient_category == 'above':
		WABOVE[DERIV_L][i_ind,j_ind] = y
	elif ref.ndim == 2 and gradient_category == 'under':
		if gradient_weights:
			WUNDER[DERIV_L][i_ind,j_ind] = y
		else:
			BUNDER[DERIV_L][i_ind,j_ind] = y
	elif ref.ndim == 2 and gradient_category == 'read':
		if gradient_weights:
			WR[DERIV_L][i_ind,j_ind] = y
		else:
			BR[DERIV_L][i_ind,j_ind] = y
	elif gradient_category == 'read':
		if gradient_weights:
			WR[DERIV_L][i_ind,j_ind,k_ind] = y
		else:
			BR[DERIV_L][i_ind,j_ind,k_ind] = y
	elif ref.ndim == 2:
		if gradient_weights:
			WW[DERIV_L][i_ind,j_ind] = y
		else:
			BW[DERIV_L][i_ind,j_ind] = y
	else:
		if gradient_weights:
			WW[DERIV_L][i_ind,j_ind,k_ind] = y
		else:
			BW[DERIV_L][i_ind,j_ind,k_ind] = y
	##
	
	
	###
	for frame in range(1,N_FRAMES+1):
		# forward
		OR,OW,mem,read_mem,OUNDER,OABOVE = forward_pass(WUNDER, BUNDER, WR,WW,BR,BW, WABOVE, BABOVE, OR_PREV, OW_PREV, mem_prev, x[frame])
		
		# reverse (compute memory partials)
		DOW_DWW, DOW_DBW, DOW_DWUNDER, DOW_DBUNDER, DMEM_PREV_DWW, DMEM_PREV_DBW, DMEM_PREV_DWUNDER, DMEM_PREV_DBUNDER, \
		DOR_DWR, DOR_DBR, DOR_DWUNDER, DOR_DBUNDER, DOR_DWW, DOR_DBW = reverse_pass_partials(WUNDER, BUNDER, WR,WW,BR,BW, \
				OUNDER, OUNDER_PREV, OR, OR_PREV, OW_PREV, OW_PREV_PREV, \
				mem_prev, mem_prev_prev, x[frame], x[frame-1], frame, DOW_DWW, DOW_DBW, \
				DOW_DWUNDER, DOW_DBUNDER, DMEM_PREV_DWW, DMEM_PREV_DBW, DMEM_PREV_DWUNDER, DMEM_PREV_DBUNDER,\
				DOR_DWR, DOR_DBR, DOR_DWUNDER, DOR_DBUNDER, DOR_DWW, DOR_DBW)

	
		# update temporal state vars
		if frame != N_FRAMES:
			OW_PREV_PREV = copy.deepcopy(OW_PREV)
			OR_PREV = copy.deepcopy(OR); OW_PREV = copy.deepcopy(OW); OUNDER_PREV = copy.deepcopy(OUNDER)
			mem_prev_prev = copy.deepcopy(mem_prev); mem_prev = copy.deepcopy(mem)
	
	# full gradients from partials
	DWR, DBR, DWW, DBW, DWUNDER, DBUNDER, DWABOVE, DBABOVE = full_gradients(read_mem, t, mem_prev, DOR_DWR, DOR_DBR, \
				DOR_DWW, DOR_DBW, DOR_DWUNDER, DOR_DBUNDER, OR, DMEM_PREV_DWW, DMEM_PREV_DBW, \
				DMEM_PREV_DWUNDER, DMEM_PREV_DBUNDER, OABOVE, WABOVE, BABOVE)

	####
	ind_counter = 0
	read_mem_ind = ind_counter; ind_counter += 1
	t_ind = ind_counter; ind_counter += 1
	mem_prev_ind = ind_counter; ind_counter += 1
	nm.set_buffer(read_mem, read_mem_ind); read_mem_shape = read_mem.shape
	nm.set_buffer(t, t_ind); t_shape = t.shape
	nm.set_buffer(mem_prev, mem_prev_ind); mem_prev_shape = mem_prev.shape
	ind_counter, DOR_DWR_IND, DOR_DWR_SHAPE = set_list_buffer(ind_counter, DOR_DWR)
	ind_counter, DOR_DBR_IND, DOR_DBR_SHAPE = set_list_buffer(ind_counter, DOR_DBR)
	ind_counter, DOR_DWW_IND, DOR_DWW_SHAPE = set_list_buffer(ind_counter, DOR_DWW)
	ind_counter, DOR_DBW_IND, DOR_DBW_SHAPE = set_list_buffer(ind_counter, DOR_DBW)
	ind_counter, DOR_DWUNDER_IND, DOR_DWUNDER_SHAPE = set_list_buffer(ind_counter, DOR_DWUNDER)
	ind_counter, DOR_DBUNDER_IND, DOR_DBUNDER_SHAPE = set_list_buffer(ind_counter, DOR_DBUNDER)
	ind_counter, OR_IND, OR_SHAPE = set_list_buffer(ind_counter, OR)
	ind_counter, DMEM_PREV_DWW_IND, DMEM_PREV_DWW_SHAPE = set_list_buffer(ind_counter, DMEM_PREV_DWW)
	ind_counter, DMEM_PREV_DBW_IND, DMEM_PREV_DBW_SHAPE = set_list_buffer(ind_counter, DMEM_PREV_DBW)
	ind_counter, DMEM_PREV_DWUNDER_IND, DMEM_PREV_DWUNDER_SHAPE = set_list_buffer(ind_counter, DMEM_PREV_DWUNDER)
	ind_counter, DMEM_PREV_DBUNDER_IND, DMEM_PREV_DBUNDER_SHAPE = set_list_buffer(ind_counter, DMEM_PREV_DBUNDER)
	ind_counter, OABOVE_IND, OABOVE_SHAPE = set_list_buffer(ind_counter, OABOVE)
	ind_counter, WABOVE_IND, WABOVE_SHAPE = set_list_buffer(ind_counter, WABOVE)
	ind_counter, BABOVE_IND, BABOVE_SHAPE = set_list_buffer(ind_counter, BABOVE)
	
	ind_counter, DWR_IND, DWR_SHAPE = set_list_buffer(ind_counter, WR)
	ind_counter, DBR_IND, DBR_SHAPE = set_list_buffer(ind_counter, BR)
	ind_counter, DWW_IND, DWW_SHAPE = set_list_buffer(ind_counter, WW)
	ind_counter, DBW_IND, DBW_SHAPE = set_list_buffer(ind_counter, BW)
	ind_counter, DWUNDER_IND, DWUNDER_SHAPE = set_list_buffer(ind_counter, WUNDER)
	ind_counter, DBUNDER_IND, DBUNDER_SHAPE = set_list_buffer(ind_counter, BUNDER)
	ind_counter, DWABOVE_IND, DWABOVE_SHAPE = set_list_buffer(ind_counter, WABOVE)
	ind_counter, DBABOVE_IND, DBABOVE_SHAPE = set_list_buffer(ind_counter, BABOVE)
	####
	
	full_gradients_gpu(read_mem_ind, t_ind, mem_prev_ind, DOR_DWR_IND, DOR_DBR_IND, \
				DOR_DWW_IND, DOR_DBW_IND, DOR_DWUNDER_IND, DOR_DBUNDER_IND, OR_IND, \
				DMEM_PREV_DWW_IND, DMEM_PREV_DBW_IND, \
				DMEM_PREV_DWUNDER_IND, DMEM_PREV_DBUNDER_IND, OABOVE_IND, WABOVE_IND, BABOVE_IND,\
				DWR_IND, DBR_IND, DWW_IND, DBW_IND, DWUNDER_IND, \
				DBUNDER_IND, DWABOVE_IND, DBABOVE_IND, \
				read_mem_shape, t_shape, mem_prev_shape, DOR_DWR_SHAPE, DOR_DBR_SHAPE, \
				DOR_DWW_SHAPE, DOR_DBW_SHAPE, DOR_DWUNDER_SHAPE, DOR_DBUNDER_SHAPE, OR_SHAPE, \
				DMEM_PREV_DWW_SHAPE, DMEM_PREV_DBW_SHAPE, \
				DMEM_PREV_DWUNDER_SHAPE, DMEM_PREV_DBUNDER_SHAPE, \
				OABOVE_SHAPE, WABOVE_SHAPE, BABOVE_SHAPE,\
				DWR_SHAPE, DBR_SHAPE, DWW_SHAPE, DBW_SHAPE, DWUNDER_SHAPE, DBUNDER_SHAPE, DWABOVE_SHAPE, DBABOVE_SHAPE, ind_counter)
	
	####
	if gradient_category == 'above':
		return DWABOVE[DERIV_L][i_ind,j_ind]
	elif ref.ndim == 2 and gradient_category == 'under':
		if gradient_weights:
			return DWUNDER[DERIV_L][i_ind,j_ind]
		else:
			return DBUNDER[DERIV_L][i_ind,j_ind]
	elif ref.ndim == 2 and gradient_category == 'read':
		if gradient_weights:
			return DWR[DERIV_L][i_ind,j_ind]
		else:
			return DBR[DERIV_L][i_ind,j_ind]
	elif gradient_category == 'read':
		if gradient_weights:
			return DWR[DERIV_L][i_ind,j_ind,k_ind]
		else:
			return DBR[DERIV_L][i_ind,j_ind,k_ind]
	elif ref.ndim == 2:
		if gradient_weights:
			return DWW[DERIV_L][i_ind,j_ind]
		else:
			return DBW[DERIV_L][i_ind,j_ind]
	else:
		if gradient_weights:
			return DWW[DERIV_L][i_ind,j_ind,k_ind]
		else:
			return DBW[DERIV_L][i_ind,j_ind,k_ind]
	
np.random.seed(np.int64(time.time()))
eps = np.sqrt(np.finfo(np.float).eps)*1e7

N_SAMPLES = 25
ratios = np.zeros(N_SAMPLES)
for sample in range(N_SAMPLES):
	i_ind = np.random.randint(ref.shape[0])
	j_ind = np.random.randint(ref.shape[1])
	if ref.ndim == 2:
		y = -1e0*ref[i_ind,j_ind]
	else:
		k_ind = np.random.randint(ref.shape[2])
		y = -1e0*ref[i_ind,j_ind,k_ind]
	
	gt = g(y); gtx = scipy.optimize.approx_fprime(np.ones(1)*y, f, eps)
	
	if gtx == 0:
		ratios[sample] = 1
	else:
		ratios[sample] = gtx/gt
	print gt, gtx, ratios[sample]
	
print ratios.mean(), ratios.std()
