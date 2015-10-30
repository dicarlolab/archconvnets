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

save_name = 'ntm_test'
n_saves = 0

WW = copy.deepcopy(WWi); WR = copy.deepcopy(WRi);
BW = copy.deepcopy(BWi); BR = copy.deepcopy(BRi);
WUNDER = copy.deepcopy(WUNDERi); BUNDER = copy.deepcopy(BUNDERi);
WABOVE = copy.deepcopy(WABOVEi); BABOVE = copy.deepcopy(BABOVEi);
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

EPS_BR = EPS_WW = EPS_BW = EPS_WR = -5e-4
EPS_BUNDER = -1e-8
EPS_WUNDER = -1e-8

SAVE_FREQ = 200
WRITE_FREQ = 2000
STOP_POINT = np.inf #200*1000*3*10 #1250*300

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
	OR,OW,mem,read_mem,OUNDER,OABOVE = forward_pass(WUNDER, BUNDER, WR,WW,BR,BW, WABOVE,BABOVE,OR_PREV, OW_PREV, mem_prev, inputs)
	
	err += np.sum((target - OUNDER[F_UNDER])**2)
	
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


	# update temporal state vars
	if frame != N_FRAMES:
		OW_PREV_PREV = copy.deepcopy(OW_PREV)
		OR_PREV = copy.deepcopy(OR); OW_PREV = copy.deepcopy(OW); OUNDER_PREV = copy.deepcopy(OUNDER)
		mem_prev_prev = copy.deepcopy(mem_prev); mem_prev = copy.deepcopy(mem)
	
	# full gradients from partials
	DWR, DBR, DWW, DBW, DWUNDER, DBUNDER, DABOVE = full_gradients(read_mem, t, mem_prev, DOR_DWR, DOR_DBR, \
			DOR_DWW, DOR_DBW, DOR_DWUNDER, DOR_DBUNDER, OR, DMEM_PREV_DWW, DMEM_PREV_DBW, \
			DMEM_PREV_DWUNDER, DMEM_PREV_DBUNDER, OABOVE, WABOVE)
	
	# take step
	if frame < STOP_POINT and frame > SAVE_FREQ:
		DWUNDER[L2_UNDER][np.isnan(DWUNDER[L2_UNDER])] = 0
		
		WR = pointwise_mult_partials_add__layers(WR, DWR, EPS_WR)
		BR = pointwise_mult_partials_add__layers(BR, DBR, EPS_BR)
		WW = pointwise_mult_partials_add__layers(WW, DWW, EPS_WW)
		BW = pointwise_mult_partials_add__layers(BW, DBW, EPS_BW)
		WUNDER = pointwise_mult_partials_add__layers(WUNDER, DWUNDER, EPS_WUNDER)
		BUNDER = pointwise_mult_partials_add__layers(BUNDER, DBUNDER, EPS_BUNDER)
	
	# print
	if frame % SAVE_FREQ == 0 and frame != 0:
		print 'err: ', err / SAVE_FREQ, 'frame: ', frame, 'time: ', time.time() - t_start, save_name

		PRINT_KEYS = [L1_UNDER, L2_UNDER, F_UNDER, KEY, BETA, IN_GATE, SHIFT, GAMMA]
		print_names = ['L1_UNDER', 'L2_UNDER', 'F_UNDER', 'KEY', 'BETA', 'IN_GATE', 'SHIFT', 'GAMMA']
		print_under = np.zeros(len(print_names)); print_under[:3] = 1
		
		max_print_len = 0
		for i in range(len(print_names)):
			if len(print_names[i]) > max_print_len:
				max_print_len = len(print_names[i])
		
		for i in range(len(print_names)):
			if print_under[i] == 1:
				print '  ', print_names[i], ' '*(max_print_len - len(print_names[i])), \
					' W: %.1e %.1e (%.1e)  B: %.1e %.1e (%.1e)' % (\
				np.min(WUNDER[PRINT_KEYS[i]]), np.max(WUNDER[PRINT_KEYS[i]]), -EPS_WUNDER*np.median(np.abs(DWUNDER[PRINT_KEYS[i]]/WUNDER[PRINT_KEYS[i]])), \
				np.min(BUNDER[PRINT_KEYS[i]]), np.max(BUNDER[PRINT_KEYS[i]]), -EPS_BUNDER*np.median(np.abs(DBUNDER[PRINT_KEYS[i]]/BUNDER[PRINT_KEYS[i]])))
			else:
				print '  ', print_names[i], ' '*(max_print_len - len(print_names[i])), \
					' WR: %.1e %.1e (%.1e)  BR: %.1e %.1e (%.1e)  WW: %.1e %.1e (%.1e)  BW: %.1e %.1e (%.1e)' % (\
					np.min(WR[PRINT_KEYS[i]]), np.max(WR[PRINT_KEYS[i]]), -EPS_WR*np.median(np.abs(DWR[PRINT_KEYS[i]]/WR[PRINT_KEYS[i]])), \
					np.min(BR[PRINT_KEYS[i]]), np.max(BR[PRINT_KEYS[i]]), -EPS_BR*np.median(np.abs(DBR[PRINT_KEYS[i]]/BR[PRINT_KEYS[i]])), \
					np.min(WW[PRINT_KEYS[i]]), np.max(WW[PRINT_KEYS[i]]), -EPS_WW*np.median(np.abs(DWW[PRINT_KEYS[i]]/WW[PRINT_KEYS[i]])), \
					np.min(BW[PRINT_KEYS[i]]), np.max(BW[PRINT_KEYS[i]]), -EPS_BW*np.median(np.abs(DBW[PRINT_KEYS[i]]/BW[PRINT_KEYS[i]])))
		print	
		
		
		err_log.append(err / SAVE_FREQ)
		err = 0
		
		savemat('/home/darren/' + save_name + '.mat', {'output_buffer': output_buffer, 'target_buffer': target_buffer, 'err_log': err_log, 'corr_buffer': corr_buffer, 'train_buffer': train_buffer, 'EPS_BR': EPS_BR, 'EPS_WW': EPS_WW, 'EPS_WR': EPS_WR, 'EPS_BUNDER': EPS_BUNDER, 'EPS_WUNDER': EPS_WUNDER})
		
		t_start = time.time()
	
	# write
	if frame % WRITE_FREQ == 0:
		print 'writing', save_name
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


