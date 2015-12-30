import time
import numpy as np
from scipy.io import savemat, loadmat
import copy
from scipy.stats import zscore
import random
import scipy
from ntm_gradients import *
from init_vars import *
from ntm_core import *

t_start = time.time()
##### which gradients to test
#DERIV_L = L1_UNDER
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
	
print ratios.mean(), ratios.std()
print time.time() - t_start
