import time
import numpy as np
from scipy.io import savemat, loadmat
import copy
from scipy.stats import zscore, pearsonr
import random
import scipy
import pickle as pk
from ntm_gradients import *
from init_vars import *
from ntm_core import *

save_name = 'ntm_test_reset_slower_learning2_faster_mem'
n_saves = 0

WW = copy.deepcopy(WWi); WR = copy.deepcopy(WRi);
BW = copy.deepcopy(BWi); BR = copy.deepcopy(BRi);
WUNDER = copy.deepcopy(WUNDERi); BUNDER = copy.deepcopy(BUNDERi);
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

EPS_BR = EPS_WW = EPS_BW = EPS_WR = -5e-3
EPS_BUNDER = -1e-7
EPS_WUNDER = -1e-7

SAVE_FREQ = 200
WRITE_FREQ = 2000
STOP_POINT = 200*1000*3*10 #1250*300

training = 0
time_length = 3
elapsed_time = 1000
frame = 0
err = 0

START_SIGNAL = 0; TRAIN_SIGNAL = 1

#target_seq = np.random.randint(2,size=time_length) - .5
target_seq = np.random.normal(size=time_length)
output_seq = np.zeros_like(target_seq)

inputs[START_SIGNAL] = 1 # start signal

train_buffer = np.zeros(SAVE_FREQ)
target_buffer = np.zeros(SAVE_FREQ)
output_buffer = np.zeros(SAVE_FREQ)
corr_buffer = np.zeros(SAVE_FREQ)
err_log = []

t_start = time.time()
while True:

	inputs[START_SIGNAL] = 0
	
	# switch from training to testing or conversely
	if elapsed_time >= time_length:
		training = 1 - training
		elapsed_time = 0
		if training == 1: # new training sequence
			
			DOR_DWR = copy.deepcopy(DOR_DWRi); DOW_DWW = copy.deepcopy(DOW_DWWi); DOR_DWW = copy.deepcopy(DOR_DWWi)
			DOR_DBR = copy.deepcopy(DOR_DBRi); DOW_DBW = copy.deepcopy(DOW_DBWi); DOR_DBW = copy.deepcopy(DOR_DBWi)
			DOW_DWUNDER = copy.deepcopy(DOW_DWUNDERi); DOR_DWUNDER = copy.deepcopy(DOR_DWUNDERi)
			DOW_DBUNDER = copy.deepcopy(DOW_DBUNDERi); DOR_DBUNDER = copy.deepcopy(DOR_DBUNDERi)
			mem_prev = copy.deepcopy(mem_previ); mem_prev_prev = copy.deepcopy(mem_previ)
			DMEM_PREV_DWW = copy.deepcopy(DMEM_PREV_DWWi); DMEM_PREV_DBW = copy.deepcopy(DMEM_PREV_DBWi)
			DMEM_PREV_DWUNDER = copy.deepcopy(DMEM_PREV_DWUNDERi); DMEM_PREV_DBUNDER = copy.deepcopy(DMEM_PREV_DBUNDERi)
			
			corr_buffer[frame % SAVE_FREQ] = pearsonr(output_seq, target_seq)[0]
			
			#target_seq = np.random.randint(2,size=time_length) - .5
			target_seq = np.random.normal(size=time_length)
			inputs[START_SIGNAL] = 1
			
	
	
	if training == 1: # train period
		inputs[TRAIN_SIGNAL] = target_seq[elapsed_time]
		target = 0
	else: # test period
		inputs[TRAIN_SIGNAL] = 0
		target = target_seq[elapsed_time]

	# forward
	OR, OW, mem, read_mem, OUNDER = forward_pass(WUNDER, BUNDER, WR,WW,BR,BW, OR_PREV, OW_PREV, mem_prev, inputs)
	
	err += np.sum((target - read_mem)**2)
	
	train_buffer[frame % SAVE_FREQ] = copy.deepcopy(inputs[TRAIN_SIGNAL])
	target_buffer[frame % SAVE_FREQ] = copy.deepcopy(target)
	output_buffer[frame % SAVE_FREQ] = read_mem.sum()
	output_seq[elapsed_time] = read_mem.sum()
	
	# reverse (compute memory partials)
	DOW_DWW, DOW_DBW, DOW_DWUNDER, DOW_DBUNDER, DMEM_PREV_DWW, DMEM_PREV_DBW, DMEM_PREV_DWUNDER, DMEM_PREV_DBUNDER, \
	DOR_DWR, DOR_DBR, DOR_DWUNDER, DOR_DBUNDER, DOR_DWW, DOR_DBW = reverse_pass_partials(WUNDER, BUNDER, WR,WW,BR,BW, \
			OUNDER, OUNDER_PREV, OR, OR_PREV, OW_PREV, OW_PREV_PREV, \
			mem_prev, mem_prev_prev, inputs, inputs_prev, frame, DOW_DWW, DOW_DBW, \
			DOW_DWUNDER, DOW_DBUNDER, DMEM_PREV_DWW, DMEM_PREV_DBW, DMEM_PREV_DWUNDER, DMEM_PREV_DBUNDER,\
			DOR_DWR, DOR_DBR, DOR_DWUNDER, DOR_DBUNDER, DOR_DWW, DOR_DBW)

	#print 'fin reverse_pass_partials'
	
	# update temporal state vars
	if frame != N_FRAMES:
		OW_PREV_PREV = copy.deepcopy(OW_PREV)
		OR_PREV = copy.deepcopy(OR); OW_PREV = copy.deepcopy(OW); OUNDER_PREV = copy.deepcopy(OUNDER)
		mem_prev_prev = copy.deepcopy(mem_prev); mem_prev = copy.deepcopy(mem)
	
	# full gradients from partials
	DWR, DBR, DWW, DBW, DWUNDER, DBUNDER = full_gradients(read_mem, target, mem_prev, DOR_DWR, DOR_DBR, \
			DOR_DWW, DOR_DBW, DOR_DWUNDER, DOR_DBUNDER, OR, DMEM_PREV_DWW, DMEM_PREV_DBW, DMEM_PREV_DWUNDER, DMEM_PREV_DBUNDER)
	
	# take step
	if frame < STOP_POINT:
		DWUNDER[L2_UNDER][np.isnan(DWUNDER[L2_UNDER])] = 0
		
		WR = pointwise_mult_partials_add__layers(WR, DWR, EPS_WR)
		BR = pointwise_mult_partials_add__layers(BR, DBR, EPS_BR)
		WW = pointwise_mult_partials_add__layers(WW, DWW, EPS_WW)
		BW = pointwise_mult_partials_add__layers(BW, DBW, EPS_BW)
		WUNDER = pointwise_mult_partials_add__layers(WUNDER, DWUNDER, EPS_WUNDER)
		BUNDER = pointwise_mult_partials_add__layers(BUNDER, DBUNDER, EPS_BUNDER)
	
	# print
	if frame % SAVE_FREQ == 0 and frame != 0:
		print err / SAVE_FREQ, frame, EPS_WR*np.median(DWR[IN_GATE]/WR[IN_GATE]), EPS_BR*np.median(DBR[IN_GATE]/BR[IN_GATE]), EPS_WW*np.median(DWW[IN_GATE]/WW[IN_GATE]), EPS_BW*np.median(DBW[IN_GATE]/BW[IN_GATE]), EPS_BUNDER*np.median(DBUNDER[F_UNDER]/BUNDER[F_UNDER]), time.time() - t_start, save_name
		
		err_log.append(err / SAVE_FREQ)
		err = 0
		
		savemat('/home/darren/' + save_name + '.mat', {'output_buffer': output_buffer, 'target_buffer': target_buffer, 'err_log': err_log, 'corr_buffer': corr_buffer, 'train_buffer': train_buffer})
		
		t_start = time.time()
	
	# write
	if frame % WRITE_FREQ == 0:
		print 'writing'
		file = open('/home/darren/ntm_saves/' + save_name + '_' + str(n_saves) + '.pk','w')
		pk.dump({'WR': WR, 'BR': BR, 'WW': WW, 'BW': BW, 'WUNDER': WUNDER, 'BUNDER': BUNDER, 'frame': frame, \
			'EPS_BR': EPS_BR, 'EPS_WW': EPS_WW, 'EPS_WR': EPS_WR, 'EPS_BUNDER': EPS_BUNDER, 'EPS_WUNDER': EPS_WUNDER}, file)
		file.close()
		
		n_saves += 1
		
		
	frame += 1
	elapsed_time += 1
	if frame == STOP_POINT:
		print 'stopping'
	if frame == (STOP_POINT + 3*SAVE_FREQ):
		break


