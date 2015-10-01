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

OR_PREV = copy.deepcopy(OR_PREVi); OW_PREV = copy.deepcopy(OW_PREVi)
OW_PREV_PREV = copy.deepcopy(OW_PREV_PREVi); OUNDER_PREV = copy.deepcopy(OUNDER_PREVi)
DOR_DWR = copy.deepcopy(DOR_DWRi); DOW_DWW = copy.deepcopy(DOW_DWWi); DOR_DWW = copy.deepcopy(DOR_DWWi)
DOR_DBR = copy.deepcopy(DOR_DBRi); DOW_DBW = copy.deepcopy(DOW_DBWi); DOR_DBW = copy.deepcopy(DOR_DBWi)
DOW_DWUNDER = copy.deepcopy(DOW_DWUNDERi); DOR_DWUNDER = copy.deepcopy(DOR_DWUNDERi)
DOW_DBUNDER = copy.deepcopy(DOW_DBUNDERi); DOR_DBUNDER = copy.deepcopy(DOR_DBUNDERi)
mem_prev = copy.deepcopy(mem_previ); mem_prev_prev = copy.deepcopy(mem_previ)
DMEM_PREV_DWW = copy.deepcopy(DMEM_PREV_DWWi); DMEM_PREV_DBW = copy.deepcopy(DMEM_PREV_DBWi)
DMEM_PREV_DWUNDER = copy.deepcopy(DMEM_PREV_DWUNDERi); DMEM_PREV_DBUNDER = copy.deepcopy(DMEM_PREV_DBUNDERi)

inputs_prev = np.zeros((2,1))
inputs = np.zeros((2,1))

EPS = 1e-20
MAX_TIME = 4
SAVE_FREQ = 50

time_length = np.random.randint(MAX_TIME-1) + 1
elapsed_time = 0
p_start = .1
frame = 0
err = 0

t_start = time.time()
while True:
	if random.random() < p_start and elapsed_time > time_length:
		time_length = np.random.randint(6)
		elapsed_time = 0
		inputs[0] = 1
		inputs[1] = time_length/3.
	else:
		inputs[0] = 0
		
	target = 1 - (time_length < elapsed_time)
	
	
	# forward
	OR, OW, mem, read_mem, OUNDER = forward_pass(WUNDER, BUNDER, WR,WW,BR,BW, OR_PREV, OW_PREV, mem_prev, inputs)
	
	err += np.sum((target - read_mem)**2)
	
	# reverse (compute memory partials)
	DOW_DWW, DOW_DBW, DOW_DWUNDER, DOW_DBUNDER, DMEM_PREV_DWW, DMEM_PREV_DBW, DMEM_PREV_DWUNDER, DMEM_PREV_DBUNDER, \
	DOR_DWR, DOR_DBR, DOR_DWUNDER, DOR_DBUNDER, DOR_DWW, DOR_DBW = reverse_pass_partials(WUNDER, BUNDER, WR,WW,BR,BW, \
			OUNDER, OUNDER_PREV, OR, OR_PREV, OW_PREV, OW_PREV_PREV, \
			mem_prev, mem_prev_prev, inputs, inputs_prev, frame, DOW_DWW, DOW_DBW, \
			DOW_DWUNDER, DOW_DBUNDER, DMEM_PREV_DWW, DMEM_PREV_DBW, DMEM_PREV_DWUNDER, DMEM_PREV_DBUNDER,\
			DOR_DWR, DOR_DBR, DOR_DWUNDER, DOR_DBUNDER, DOR_DWW, DOR_DBW)


	# update temporal state vars
	if frame != N_FRAMES:
		OW_PREV_PREV = copy.deepcopy(OW_PREV)
		OR_PREV = copy.deepcopy(OR); OW_PREV = copy.deepcopy(OW); OUNDER_PREV = copy.deepcopy(OUNDER)
		mem_prev_prev = copy.deepcopy(mem_prev); mem_prev = copy.deepcopy(mem)
	
	# full gradients from partials
	DWR, DBR, DWW, DBW, DWUNDER, DBUNDER = full_gradients(read_mem, target, mem_prev, DOR_DWR, DOR_DBR, \
			DOR_DWW, DOR_DBW, DOR_DWUNDER, DOR_DBUNDER, OR, DMEM_PREV_DWW, DMEM_PREV_DBW, DMEM_PREV_DWUNDER, DMEM_PREV_DBUNDER)
	
	# take step
	WR = pointwise_mult_partials_add__layers(WR, DWR, EPS)
	BR = pointwise_mult_partials_add__layers(BR, DBR, EPS)
	WW = pointwise_mult_partials_add__layers(WW, DWW, EPS)
	BW = pointwise_mult_partials_add__layers(BW, DBW, EPS)
	WUNDER = pointwise_mult_partials_add__layers(WUNDER, DWUNDER, EPS)
	BUNDER = pointwise_mult_partials_add__layers(BUNDER, DBUNDER, EPS)
	
	if frame % SAVE_FREQ == 0:
		print err / SAVE_FREQ, time.time() - t_start
		err = 0
		t_start = time.time()
	frame += 1


