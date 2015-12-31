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
from archconvnets.unsupervised.ntm_module.ntm_module import init_buffer, set_list_buffer, return_list_buffer, return_buffer

t_copy = 0
t_start = time.time()
##### which gradients to test
#DERIV_L = L1_UNDER #### double-check!
#DERIV_L = L2_UNDER
#DERIV_L = F_UNDER

#DERIV_L = L1_ABOVE
#DERIV_L = F_ABOVE

#DERIV_L = SHIFT
#DERIV_L = IN_GATE
#DERIV_L = KEY
#DERIV_L = BETA
DERIV_L = ADD
#DERIV_L = ERASE
#DERIV_L = GAMMA

gradient_category = 'write'
#gradient_category = 'read'
#gradient_category = 'under'
#gradient_category = 'above'

#gradient_weights = False # false means bias terms
gradient_weights = True

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
	global t_copy
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
		
		nm.free_all_buffers()
		
		t_copy_start = time.time()
		#### put data on gpu
		READ_MEM = init_buffer(read_mem)
		T = init_buffer(t)
		MEM = init_buffer(mem)
		MEM_PREV = init_buffer(mem_prev)
		
		L_OW = set_list_buffer(OW)
		
		L_DOR_DWR = set_list_buffer(DOR_DWR)
		L_DOR_DBR = set_list_buffer(DOR_DBR)
		L_DOR_DWW = set_list_buffer(DOR_DWW)
		L_DOR_DBW = set_list_buffer(DOR_DBW)
		L_DOR_DWUNDER = set_list_buffer(DOR_DWUNDER)
		L_DOR_DBUNDER = set_list_buffer(DOR_DBUNDER)
		L_OR = set_list_buffer(OR)
		L_DMEM_PREV_DWW = set_list_buffer(DMEM_PREV_DWW)
		L_DMEM_PREV_DBW = set_list_buffer(DMEM_PREV_DBW)
		L_DMEM_PREV_DWUNDER = set_list_buffer(DMEM_PREV_DWUNDER)
		L_DMEM_PREV_DBUNDER = set_list_buffer(DMEM_PREV_DBUNDER)
		L_OABOVE = set_list_buffer(OABOVE)
		L_WABOVE = set_list_buffer(WABOVE)
		L_BABOVE = set_list_buffer(BABOVE)
		
		L_WR = set_list_buffer(WR)
		L_BR = set_list_buffer(BR)
		L_WW = set_list_buffer(WW)
		L_BW = set_list_buffer(BW)
		L_WUNDER = set_list_buffer(WUNDER)
		L_BUNDER = set_list_buffer(BUNDER)
		L_WABOVE = set_list_buffer(WABOVE)
		L_BABOVE = set_list_buffer(BABOVE)
		
		L_OUNDER = set_list_buffer(OUNDER)
		L_OUNDER_PREV = set_list_buffer(OUNDER_PREV)
		L_OR_PREV = set_list_buffer(OR_PREV)
		L_OW_PREV = set_list_buffer(OW_PREV)
		L_OW_PREV_PREV = set_list_buffer(OW_PREV_PREV)
		MEM_PREV = init_buffer(mem_prev)
		MEM_PREV_PREV = init_buffer(mem_prev_prev)
		X = init_buffer(x[frame])
		X_PREV = init_buffer(x[frame-1])
		
		L_DOW_DWW = set_list_buffer(DOW_DWW)
		L_DOW_DBW = set_list_buffer(DOW_DBW)
		L_DOW_DWUNDER = set_list_buffer(DOW_DWUNDER)
		L_DOW_DBUNDER = set_list_buffer(DOW_DBUNDER)
		
		t_copy += time.time() - t_copy_start
		####
		
		# reverse (compute memory partials)
		L_DOW_DWW, L_DOW_DBW, L_DOW_DWUNDER, L_DOW_DBUNDER, L_DMEM_PREV_DWW, L_DMEM_PREV_DBW, L_DMEM_PREV_DWUNDER, L_DMEM_PREV_DBUNDER, L_DOR_DWR, L_DOR_DBR, L_DOR_DWUNDER, L_DOR_DBUNDER, L_DOR_DWW, L_DOR_DBW = \
			reverse_pass_partials(L_WUNDER, L_BUNDER, L_WR,L_WW,L_BR,L_BW, L_OUNDER, L_OUNDER_PREV, L_OR, L_OR_PREV, L_OW_PREV, L_OW_PREV_PREV, MEM_PREV, MEM_PREV_PREV, X, X_PREV, frame, L_DOW_DWW, L_DOW_DBW, L_DOW_DWUNDER, L_DOW_DBUNDER, L_DMEM_PREV_DWW, L_DMEM_PREV_DBW, L_DMEM_PREV_DWUNDER, L_DMEM_PREV_DBUNDER, L_DOR_DWR, L_DOR_DBR, L_DOR_DWUNDER, L_DOR_DBUNDER, L_DOR_DWW, L_DOR_DBW)

		nm.sync()
		# update temporal state vars
		if frame != N_FRAMES:
			t_copy_start = time.time()
			OW_PREV_PREV = return_list_buffer(L_OW_PREV)
			OR_PREV = return_list_buffer(L_OR); OW_PREV = return_list_buffer(L_OW); OUNDER_PREV = return_list_buffer(L_OUNDER)
			mem_prev_prev = return_buffer(MEM_PREV); mem_prev = return_buffer(MEM)
			t_copy += time.time() - t_copy_start
	
	# full gradients from partials
	
	L_DWR, L_DBR, L_DWW, L_DBW, L_DWUNDER, L_DBUNDER, L_DWABOVE, L_DBABOVE = \
		full_gradients_gpu(READ_MEM, T, MEM_PREV, L_DOR_DWR, L_DOR_DBR, L_DOR_DWW, L_DOR_DBW, L_DOR_DWUNDER, L_DOR_DBUNDER, L_OR, L_DMEM_PREV_DWW, L_DMEM_PREV_DBW, L_DMEM_PREV_DWUNDER, L_DMEM_PREV_DBUNDER, L_OABOVE, L_WABOVE)
	nm.sync()
	t_copy_start = time.time()
	DWR = return_list_buffer(L_DWR, L_WR)
	DBR = return_list_buffer(L_DBR, L_BR)
	DWW = return_list_buffer(L_DWW, L_WW)
	DBW = return_list_buffer(L_DBW, L_BW)
	DWUNDER = return_list_buffer(L_DWUNDER, L_WUNDER)
	DBUNDER = return_list_buffer(L_DBUNDER, L_BUNDER)
	DWABOVE = return_list_buffer(L_DWABOVE, L_WABOVE)
	DBABOVE = return_list_buffer(L_DBABOVE, L_BABOVE)
	t_copy += time.time() - t_copy_start
	
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
eps = np.sqrt(np.finfo(np.float).eps)*1e8

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
	
inf_vals = np.abs(ratios) == np.inf
if inf_vals.sum():
	print '***', inf_vals.sum(), ' non-zeros when should be zero ***'
	ratios[inf_vals] = 0
print ratios.mean(), ratios.std()
print time.time() - t_start, t_copy, time.time() - t_start - t_copy