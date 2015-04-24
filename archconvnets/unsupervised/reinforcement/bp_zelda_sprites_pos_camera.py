from archconvnets.unsupervised.cudnn_module.cudnn_module import *
import time
import numpy as np
import numexpr as ne
from scipy.io import savemat, loadmat
import copy
from scipy.stats import zscore
import random
import gnumpy as gpu
import scipy
import PIL
import Image

SAVE_FREQ = 4000
MEM_SZ = 500000
EPS_GREED_FINAL = .1
EPS_GREED_FINAL_TIME = 5000000
GAMMA = 0.99
BATCH_SZ = 32
NETWORK_UPDATE = 10000
EPS = 2.5e-3
MOM_WEIGHT = 0.95
PLAYER_MOV_RATE = 8

IMG_SZ = 92
MAX_LOC = 92 - 10

F1_scale = 1e-2
F2_scale = 1e-2
F3_scale = 1e-2
FL_scale = 1e-2

N = 64#32
n1 = N # L1 filters
n2 = N# ...
n3 = N

s3 = 3 # L1 filter size (px)
s2 = 4 # ...
s1 = 5

N_C = 3 # directions M, L, R

file_name = '/home/darren/reinforcement_saves/reinforcement_'

max_output_sz3  = 12

GPU_CUR = 2
GPU_PREV = 3

# gpu buffer indices
MAX_OUTPUT1 = 0; DF2_DATA = 1; CONV_OUTPUT1 = 2; DPOOL1 = 3
F1_IND = 4; IMGS_PAD = 5; DF1 = 6; F2_IND = 11
D_UNPOOL2 = 12;F3_IND = 13; MAX_OUTPUT2 = 14; MAX_OUTPUT3 = 15
CONV_OUTPUT1 = 19; CONV_OUTPUT2 = 20; CONV_OUTPUT3 = 21
DF2 = 25; DPOOL2 = 26; DF3_DATA = 27; DPOOL3 = 28; DF3 = 29; FL_PRED = 30;
FL_IND = 31; PRED = 32; DFL = 33

#np.random.seed(6666)
F1 = np.single(np.random.normal(scale=F1_scale, size=(n1, 3, s1, s1)))
F2 = np.single(np.random.normal(scale=F2_scale, size=(n2, n1, s2, s2)))
F3 = np.single(np.random.normal(scale=F3_scale, size=(n3, n2, s3, s3)))
FL = np.single(np.random.normal(scale=FL_scale, size=(N_C, n1, max_output_sz3, max_output_sz3)))

FL_prev = copy.deepcopy(FL)
set_buffer(F1, F1_IND, filter_flag=1, gpu=GPU_PREV)
set_buffer(F2, F2_IND, filter_flag=1, gpu=GPU_PREV)
set_buffer(F3, F3_IND, filter_flag=1, gpu=GPU_PREV)
set_buffer(FL, FL_IND, filter_flag=1, gpu=GPU_PREV)

set_buffer(F1, F1_IND, filter_flag=1, gpu=GPU_CUR)
set_buffer(F2, F2_IND, filter_flag=1, gpu=GPU_CUR)
set_buffer(F3, F3_IND, filter_flag=1, gpu=GPU_CUR)
set_buffer(FL, FL_IND, filter_flag=1, gpu=GPU_CUR)

F1_init = copy.deepcopy(F1)

dF1 = np.zeros_like(F1)
dF2 = np.zeros_like(F2)
dF3 = np.zeros_like(F3)
dFL = np.zeros_like(FL)

dF1_mom = np.zeros_like(F1)
dF2_mom = np.zeros_like(F2)
dF3_mom = np.zeros_like(F3)
dFL_mom = np.zeros_like(FL)

r_total = 0
r_total_plot = []
network_updates = 0
step = 0
err = 0
err_plot = []

###########
# init scene
ZELDA_SZ = 23
grass_scene = np.asarray(PIL.Image.open('grass_scene.png'))

zelda = np.asarray(PIL.Image.open('zelda.png').resize((ZELDA_SZ, ZELDA_SZ)))
zelda_mask = np.asarray(PIL.Image.open('zelda_mask.png').resize((ZELDA_SZ, ZELDA_SZ)))

dog = np.asarray(PIL.Image.open('dog.png').resize((ZELDA_SZ, ZELDA_SZ)))
dog_mask = np.asarray(PIL.Image.open('dog_mask.png').resize((ZELDA_SZ, ZELDA_SZ)))

dog_flip = np.asarray(PIL.Image.open('dog.png').resize((ZELDA_SZ, ZELDA_SZ)))[:,::-1]
dog_flip_mask = np.asarray(PIL.Image.open('dog_mask.png').resize((ZELDA_SZ, ZELDA_SZ)))[:,::-1]

heart = np.asarray(PIL.Image.open('heart.png').resize((ZELDA_SZ, ZELDA_SZ)))
heart_mask = np.asarray(PIL.Image.open('heart_mask.png').resize((ZELDA_SZ, ZELDA_SZ)))

zelda_mask_inds = np.nonzero(255-zelda_mask.ravel())[0]

###########todo mean img

PLAYER_MOV_RATE = 10
PLAYER_ROT_RATE = 2.*np.pi/8.

ZELDA_SZ_H = ZELDA_SZ/2

IMG_SZ = 92
IMG_SZ_H = IMG_SZ / 2

IMG_SZ_W = 130
IMG_SZ_WH = IMG_SZ_W / 2

PAD = (IMG_SZ_W - IMG_SZ) / 2.

ROOM_SZ = 1024

######## initial positions
N_DOGS = 200
N_HEARTS = 200
dogs = np.random.randint(ROOM_SZ-ZELDA_SZ, size=(2,N_DOGS))
dog_show_flip = np.random.randint(2, size=N_DOGS)
hearts = np.random.randint(ROOM_SZ-ZELDA_SZ, size=(2,N_HEARTS))
player = np.random.randint(IMG_SZ_W, ROOM_SZ-IMG_SZ_W, size=2)
rotate = 0.

######### buffers for replay
dogs_show_flip_input = np.zeros((MEM_SZ, N_DOGS), dtype='int')

dogs_input = np.zeros((MEM_SZ, 2, N_DOGS), dtype='int')
hearts_input = np.zeros((MEM_SZ, 2, N_HEARTS), dtype='int')
player_input = np.zeros((MEM_SZ, 2), dtype='int')
action_input = np.zeros(MEM_SZ, dtype='int')
rotate_input = np.zeros(MEM_SZ)

r_output = np.zeros(MEM_SZ)
y_outputs = np.zeros(MEM_SZ)
y_network_ver = -np.ones(MEM_SZ)
dogs_output = np.zeros((MEM_SZ, 2, N_DOGS), dtype='int')
hearts_output = np.zeros((MEM_SZ, 2, N_HEARTS), dtype='int')
player_output = np.zeros((MEM_SZ, 2), dtype='int')
rotate_output = np.zeros(MEM_SZ)

dogs_recent = np.zeros((SAVE_FREQ, 2, N_DOGS), dtype='int')
dog_show_flip_recent = np.zeros((SAVE_FREQ, N_DOGS), dtype='int')
hearts_recent = np.zeros((SAVE_FREQ, 2, N_DOGS), dtype='int')
player_recent = np.zeros((SAVE_FREQ, 2), dtype='int')
action_recent = np.zeros(SAVE_FREQ, dtype='int')
rotate_recent = np.zeros(SAVE_FREQ)
r_recent = np.zeros(SAVE_FREQ)

####################################################################

def render_scene(player, dogs, hearts, rotate_radians):
    dogs_in_room = np.nonzero( (dogs[0] >= (player[0]-IMG_SZ_WH)) * (dogs[1] >= (player[1]-IMG_SZ_WH)) * \
                           (dogs[0] <= (player[0]+IMG_SZ_WH)) * (dogs[1] <= (player[1]+IMG_SZ_WH)))[0]
    hearts_in_room = np.nonzero( (hearts[0] >= (player[0]-IMG_SZ_WH)) * (hearts[1] >= (player[1]-IMG_SZ_WH)) * \
                           (hearts[0] <= (player[0]+IMG_SZ_WH)) * (hearts[1] <= (player[1]+IMG_SZ_WH)))[0]
    n_dogs_in_room = len(dogs_in_room)
    n_hearts_in_room = len(hearts_in_room)
    
    w = copy.deepcopy(grass_scene[player[1]-IMG_SZ_WH:player[1]+IMG_SZ_WH, player[0]-IMG_SZ_WH:player[0]+IMG_SZ_WH])
        
    w = copy.deepcopy(np.asarray(Image.fromarray(w).rotate(180*rotate_radians/np.pi)))
    
    w = w[PAD:PAD+IMG_SZ, PAD:PAD+IMG_SZ]
    
    ########### place hearts
    for heart_i in range(n_hearts_in_room):
        heart_x = hearts[0, hearts_in_room[heart_i]]
        heart_y = hearts[1, hearts_in_room[heart_i]]
        
        if heart_y >= player[1] and heart_x == player[0]:
            theta = np.pi/2
        elif heart_y < player[1] and heart_x == player[0]:
            theta = -np.pi/2
        else:
            theta = np.arctan(np.single(heart_y - player[1])/(heart_x - player[0]))
        
        d = np.sqrt((np.single(heart_y) - player[1])**2 + (np.single(heart_x) - player[0])**2)
    
        heart_yn = np.round(d*np.sin(theta - rotate_radians))
        heart_xn = np.round(d*np.cos(theta - rotate_radians))
    
        y_min = (heart_yn)+IMG_SZ_H-ZELDA_SZ_H
        x_min = (heart_xn)+IMG_SZ_H-ZELDA_SZ_H
        
        y_start = np.max((0,y_min))
        x_start = np.max((0,x_min))
        
        oy_start = np.max((0,-y_min))
        ox_start = np.max((0,-x_min))
        
        y_end = (heart_yn)+IMG_SZ_H+ZELDA_SZ_H+1
        x_end = (heart_xn)+IMG_SZ_H+ZELDA_SZ_H+1
        
        oy_end = ZELDA_SZ - ((heart_yn)+IMG_SZ_H+ZELDA_SZ_H+1 - IMG_SZ)
        ox_end = ZELDA_SZ - ((heart_xn)+IMG_SZ_H+ZELDA_SZ_H+1 - IMG_SZ)
        
        if oy_end > 0 and ox_end > 0 and x_end > 0 and y_end > 0:
            heart_mask_inds = np.nonzero(255-heart_mask[oy_start:oy_end, ox_start:ox_end].ravel())[0]
            
            t = w[y_start:y_end, x_start:x_end].ravel()
            
            t[heart_mask_inds] = heart[oy_start:oy_end, ox_start:ox_end].ravel()[heart_mask_inds]
            
            w[y_start:y_end, x_start:x_end] = t.reshape(w[y_start:y_end, x_start:x_end].shape)
    
    ############### place dogs
    for dog_i in range(n_dogs_in_room):
        dog_x = dogs[0, dogs_in_room[dog_i]]
        dog_y = dogs[1, dogs_in_room[dog_i]]
        
        if dog_y >= player[1] and dog_x == player[0]:
            theta = np.pi/2
        elif dog_y < player[1] and dog_x == player[0]:
            theta = -np.pi/2
        else:
            theta = np.arctan(np.single(dog_y - player[1])/(dog_x - player[0]))
        
        d = np.sqrt((np.single(dog_y) - player[1])**2 + (np.single(dog_x) - player[0])**2)
    
        dog_yn = np.round(d*np.sin(theta - rotate_radians))
        dog_xn = np.round(d*np.cos(theta - rotate_radians))
    
        y_min = (dog_yn)+IMG_SZ_H-ZELDA_SZ_H
        x_min = (dog_xn)+IMG_SZ_H-ZELDA_SZ_H
        
        y_start = np.max((0,y_min))
        x_start = np.max((0,x_min))
        
        oy_start = np.max((0,-y_min))
        ox_start = np.max((0,-x_min))
        
        y_end = (dog_yn)+IMG_SZ_H+ZELDA_SZ_H+1
        x_end = (dog_xn)+IMG_SZ_H+ZELDA_SZ_H+1
        
        oy_end = ZELDA_SZ - ((dog_yn)+IMG_SZ_H+ZELDA_SZ_H+1 - IMG_SZ)
        ox_end = ZELDA_SZ - ((dog_xn)+IMG_SZ_H+ZELDA_SZ_H+1 - IMG_SZ)
        
        if oy_end > 0 and ox_end > 0 and x_end > 0 and y_end > 0:
            t = w[y_start:y_end, x_start:x_end].ravel()
            
            if dog_show_flip[dogs_in_room[dog_i]] == 1:
                dog_mask_inds = np.nonzero(255-dog_flip_mask[oy_start:oy_end, ox_start:ox_end].ravel())[0]
                t[dog_mask_inds] = dog_flip[oy_start:oy_end, ox_start:ox_end].ravel()[dog_mask_inds]
            else:
                dog_mask_inds = np.nonzero(255-dog_mask[oy_start:oy_end, ox_start:ox_end].ravel())[0]
                t[dog_mask_inds] = dog[oy_start:oy_end, ox_start:ox_end].ravel()[dog_mask_inds]
            
            w[y_start:y_end, x_start:x_end] = t.reshape(w[y_start:y_end, x_start:x_end].shape)
    
    # place zelda
    t = w[IMG_SZ_H-ZELDA_SZ_H:IMG_SZ_H+ZELDA_SZ_H+1, IMG_SZ_H-ZELDA_SZ_H:IMG_SZ_H+ZELDA_SZ_H+1].ravel()
    t[zelda_mask_inds] = zelda.ravel()[zelda_mask_inds]
    w[IMG_SZ_H-ZELDA_SZ_H:IMG_SZ_H+ZELDA_SZ_H+1, IMG_SZ_H-ZELDA_SZ_H:IMG_SZ_H+ZELDA_SZ_H+1] = t.reshape((ZELDA_SZ, ZELDA_SZ, 3))
    
    return np.ascontiguousarray(w.transpose((2,0,1))[np.newaxis])

####################################################################


if __name__ == "__main__":
	# mean image
	N_MEAN_SAMPLES = 5000
	mean_img = np.zeros((1,3,IMG_SZ,IMG_SZ),dtype='single')

	for i in range(N_MEAN_SAMPLES):
		dogs = np.random.randint(ROOM_SZ-ZELDA_SZ, size=(2,N_DOGS))
		dog_show_flip = np.random.randint(2, size=N_DOGS)
		hearts = np.random.randint(ROOM_SZ-ZELDA_SZ, size=(2,N_HEARTS))
		player = np.random.randint(IMG_SZ_W, ROOM_SZ-IMG_SZ_W, size=2)
		r = np.random.random()*2*np.pi
		mean_img += render_scene(player, dogs, hearts, r)
	mean_img /= N_MEAN_SAMPLES

	######

	t_start = time.time()
	while True:
		mem_loc  = step % MEM_SZ
		
		# copy current state
		dogs_input[mem_loc] = copy.deepcopy(dogs)
		hearts_input[mem_loc] = copy.deepcopy(hearts)
		rotate_input[mem_loc] = rotate
		player_input[mem_loc] = copy.deepcopy(player)
		y_network_ver[mem_loc] = -1
		
		img = render_scene(player, dogs, hearts, rotate)
		
		# choose action
		CHANCE_RAND = np.max((1 - ((1-EPS_GREED_FINAL)/EPS_GREED_FINAL_TIME)*(step - MEM_SZ), EPS_GREED_FINAL))
		if np.random.rand() <= CHANCE_RAND:
			action = np.random.randint(3)
		else:
			# forward pass
			set_buffer(img - mean_img, IMGS_PAD, gpu=GPU_CUR)
				
			conv_buffers(F1_IND, IMGS_PAD, CONV_OUTPUT1, gpu=GPU_CUR)
			max_pool_cudnn_buffers(CONV_OUTPUT1, MAX_OUTPUT1, gpu=GPU_CUR)
			conv_buffers(F2_IND, MAX_OUTPUT1, CONV_OUTPUT2, gpu=GPU_CUR)
			max_pool_cudnn_buffers(CONV_OUTPUT2, MAX_OUTPUT2, gpu=GPU_CUR)
			conv_buffers(F3_IND, MAX_OUTPUT2, CONV_OUTPUT3, gpu=GPU_CUR)
			max_pool_cudnn_buffers(CONV_OUTPUT3, MAX_OUTPUT3, gpu=GPU_CUR)
			
			max_output3 = return_buffer(MAX_OUTPUT3, gpu=GPU_CUR)
			
			pred = np.einsum(FL, range(4), max_output3, [4,1,2,3], [0])
			
			action = np.argmax(pred)

		# perform action
		if action == 0 and player[0] > PLAYER_MOV_RATE:
			dx = PLAYER_MOV_RATE * np.cos(-rotate - np.pi/2)
			dy = PLAYER_MOV_RATE * np.sin(-rotate + np.pi/2)
			if (player[0] + dx) < (ROOM_SZ-IMG_SZ_WH) and (player[1] + dy) < (ROOM_SZ-IMG_SZ_WH) and \
				(player[0] + dx) > IMG_SZ_WH and (player[1] + dy) > IMG_SZ_WH:
					player[0] += dx
					player[1] += dy
		elif action == 1:
			rotate += PLAYER_ROT_RATE
		elif action == 2:
			rotate -= PLAYER_ROT_RATE
		
		# determine reward, choose new block locations
		r = 0

		# dog collision, place new dogs
		collision = np.nonzero(((player[0] >= dogs[0]) * (player[0] <= (dogs[0]+ZELDA_SZ)) * (player[1] >= dogs[1]) * (player[1] <= (dogs[1]+ZELDA_SZ))) + \
					((player[0]+ZELDA_SZ) >= dogs[0]) * ((player[0]+ZELDA_SZ) <= (dogs[0]+ZELDA_SZ)) * (player[1] >= dogs[1]) * (player[1] <= (dogs[1]+ZELDA_SZ)) + \
					((player[0]+ZELDA_SZ) >= dogs[0]) * ((player[0]+ZELDA_SZ) <= (dogs[0]+ZELDA_SZ)) * ((player[1]+ZELDA_SZ) >= dogs[1]) * ((player[1]+ZELDA_SZ) <= (dogs[1]+ZELDA_SZ)) + \
					(player[0] >= dogs[0]) * (player[0] <= (dogs[0]+ZELDA_SZ)) * ((player[1]+ZELDA_SZ) >= dogs[1]) * ((player[1]+ZELDA_SZ) <= (dogs[1]+ZELDA_SZ)))[0]
		while len(collision) >= 1:
			r = -1
			dogs[:,collision] = np.random.randint(ROOM_SZ-ZELDA_SZ, size=(2,len(collision)))
			collision = np.nonzero(((player[0] >= dogs[0]) * (player[0] <= (dogs[0]+ZELDA_SZ)) * (player[1] >= dogs[1]) * (player[1] <= (dogs[1]+ZELDA_SZ))) + \
					((player[0]+ZELDA_SZ) >= dogs[0]) * ((player[0]+ZELDA_SZ) <= (dogs[0]+ZELDA_SZ)) * (player[1] >= dogs[1]) * (player[1] <= (dogs[1]+ZELDA_SZ)) + \
					((player[0]+ZELDA_SZ) >= dogs[0]) * ((player[0]+ZELDA_SZ) <= (dogs[0]+ZELDA_SZ)) * ((player[1]+ZELDA_SZ) >= dogs[1]) * ((player[1]+ZELDA_SZ) <= (dogs[1]+ZELDA_SZ)) + \
					(player[0] >= dogs[0]) * (player[0] <= (dogs[0]+ZELDA_SZ)) * ((player[1]+ZELDA_SZ) >= dogs[1]) * ((player[1]+ZELDA_SZ) <= (dogs[1]+ZELDA_SZ)))[0]

		# heart collision, place new heart
		collision = np.nonzero(((player[0] >= hearts[0]) * (player[0] <= (hearts[0]+ZELDA_SZ)) * (player[1] >= hearts[1]) * (player[1] <= (hearts[1]+ZELDA_SZ)))+ \
					((player[0]+ZELDA_SZ) >= hearts[0]) * ((player[0]+ZELDA_SZ) <= (hearts[0]+ZELDA_SZ)) * (player[1] >= hearts[1]) * (player[1] <= (hearts[1]+ZELDA_SZ)) + \
					((player[0]+ZELDA_SZ) >= hearts[0]) * ((player[0]+ZELDA_SZ) <= (hearts[0]+ZELDA_SZ)) * ((player[1]+ZELDA_SZ) >= hearts[1]) * ((player[1]+ZELDA_SZ) <= (hearts[1]+ZELDA_SZ)) + \
					(player[0] >= hearts[0]) * (player[0] <= (hearts[0]+ZELDA_SZ)) * ((player[1]+ZELDA_SZ) >= hearts[1]) * ((player[1]+ZELDA_SZ) <= (hearts[1]+ZELDA_SZ)))[0]
		while len(collision) >= 1:
			r = 1
			hearts[:,collision] = np.random.randint(ROOM_SZ-ZELDA_SZ, size=(2,len(collision)))
			collision = np.nonzero(((player[0] >= hearts[0]) * (player[0] <= (hearts[0]+ZELDA_SZ)) * (player[1] >= hearts[1]) * (player[1] <= (hearts[1]+ZELDA_SZ))) + \
					((player[0]+ZELDA_SZ) >= hearts[0]) * ((player[0]+ZELDA_SZ) <= (hearts[0]+ZELDA_SZ)) * (player[1] >= hearts[1]) * (player[1] <= (hearts[1]+ZELDA_SZ)) + \
					((player[0]+ZELDA_SZ) >= hearts[0]) * ((player[0]+ZELDA_SZ) <= (hearts[0]+ZELDA_SZ)) * ((player[1]+ZELDA_SZ) >= hearts[1]) * ((player[1]+ZELDA_SZ) <= (hearts[1]+ZELDA_SZ)) + \
					(player[0] >= hearts[0]) * (player[0] <= (hearts[0]+ZELDA_SZ)) * ((player[1]+ZELDA_SZ) >= hearts[1]) * ((player[1]+ZELDA_SZ) <= (hearts[1]+ZELDA_SZ)))[0]

		r_total += r
		
		# copy current state
		dogs_output[mem_loc] = copy.deepcopy(dogs)
		hearts_output[mem_loc] = copy.deepcopy(hearts)
		player_output[mem_loc] = copy.deepcopy(player)
		rotate_output[mem_loc] = rotate
		r_output[mem_loc] = r
		action_input[mem_loc] = action
		
		# debug/for visualizations
		dog_show_flip_recent[step % SAVE_FREQ] = copy.deepcopy(dog_show_flip)
		dogs_recent[step % SAVE_FREQ] = copy.deepcopy(dogs)
		hearts_recent[step % SAVE_FREQ] = copy.deepcopy(hearts)
		player_recent[step % SAVE_FREQ] = copy.deepcopy(player)
		action_recent[step % SAVE_FREQ] = action
		rotate_recent[step % SAVE_FREQ] = rotate
		r_recent[step % SAVE_FREQ] = r
		
		if step == MEM_SZ:
			print 'beginning gradient computations'
		
		######################################
		# update gradient?
		if step >= MEM_SZ:
			trans = np.random.randint(MEM_SZ)
			
			# show blocks
			img_prev = render_scene(player_output[trans], dogs_output[trans], hearts_output[trans], rotate_output[trans])
			img_cur = render_scene(player_input[trans], dogs_input[trans], hearts_input[trans], rotate_input[trans])
			
			set_buffer(img_cur - mean_img, IMGS_PAD, gpu=GPU_CUR)
			
			# forward pass current network
			conv_buffers(F1_IND, IMGS_PAD, CONV_OUTPUT1, gpu=GPU_CUR)
			max_pool_cudnn_buffers(CONV_OUTPUT1, MAX_OUTPUT1, gpu=GPU_CUR)
			conv_buffers(F2_IND, MAX_OUTPUT1, CONV_OUTPUT2, gpu=GPU_CUR)
			max_pool_cudnn_buffers(CONV_OUTPUT2, MAX_OUTPUT2, gpu=GPU_CUR)
			conv_buffers(F3_IND, MAX_OUTPUT2, CONV_OUTPUT3, gpu=GPU_CUR)
			max_pool_cudnn_buffers(CONV_OUTPUT3, MAX_OUTPUT3, gpu=GPU_CUR)
			
			# forward pass prev network
			if y_network_ver[trans] != (network_updates % NETWORK_UPDATE):
				set_buffer(img_prev - mean_img, IMGS_PAD, gpu=GPU_PREV)
			
				conv_buffers(F1_IND, IMGS_PAD, CONV_OUTPUT1, gpu=GPU_PREV)
				max_pool_cudnn_buffers(CONV_OUTPUT1, MAX_OUTPUT1, gpu=GPU_PREV)
				conv_buffers(F2_IND, MAX_OUTPUT1, CONV_OUTPUT2, gpu=GPU_PREV)
				max_pool_cudnn_buffers(CONV_OUTPUT2, MAX_OUTPUT2, gpu=GPU_PREV)
				conv_buffers(F3_IND, MAX_OUTPUT2, CONV_OUTPUT3, gpu=GPU_PREV)
				max_pool_cudnn_buffers(CONV_OUTPUT3, MAX_OUTPUT3, gpu=GPU_PREV)
			
				# compute target
				max_output3 = return_buffer(MAX_OUTPUT3, gpu=GPU_PREV)
				pred_prev = np.einsum(FL_prev, range(4), max_output3, [4,1,2,3], [0])
				y_outputs[trans] = r_output[trans] + GAMMA * np.max(pred_prev)
				y_network_ver[trans] = network_updates % NETWORK_UPDATE
			
			max_output3 = return_buffer(MAX_OUTPUT3, gpu=GPU_CUR)
			
			pred = np.einsum(FL, range(4), max_output3, [4,1,2,3], [0])
			pred_m_Y = y_outputs[trans] - pred[action_input[trans]]
			
			err += pred_m_Y**2
			
			FL_pred = np.ascontiguousarray((FL[action_input[trans]] * pred_m_Y)[np.newaxis])
			
			set_buffer(FL_pred, FL_PRED, gpu=GPU_CUR)
			
			########### backprop
			max_pool_back_cudnn_buffers(MAX_OUTPUT3, FL_PRED, CONV_OUTPUT3, DPOOL3, gpu=GPU_CUR)
			conv_dfilter_buffers(F3_IND, MAX_OUTPUT2, DPOOL3, DF3, stream=3, gpu=GPU_CUR)
			conv_ddata_buffers(F3_IND, MAX_OUTPUT2, DPOOL3, DF3_DATA, gpu=GPU_CUR)
			max_pool_back_cudnn_buffers(MAX_OUTPUT2, DF3_DATA, CONV_OUTPUT2, DPOOL2, gpu=GPU_CUR)
			conv_ddata_buffers(F2_IND, MAX_OUTPUT1, DPOOL2, DF2_DATA, gpu=GPU_CUR)
			conv_dfilter_buffers(F2_IND, MAX_OUTPUT1, DPOOL2, DF2, stream=2, gpu=GPU_CUR)
			max_pool_back_cudnn_buffers(MAX_OUTPUT1, DF2_DATA, CONV_OUTPUT1, DPOOL1, gpu=GPU_CUR)
			conv_dfilter_buffers(F1_IND, IMGS_PAD, DPOOL1, DF1, stream=1, gpu=GPU_CUR)

			### return
			dFL[action_input[trans]] += max_output3[0]*pred_m_Y
			dF3 += return_buffer(DF3, stream=3, gpu=GPU_CUR)
			dF2 += return_buffer(DF2, stream=2, gpu=GPU_CUR)
			dF1 += return_buffer(DF1, stream=1, gpu=GPU_CUR)
			
			#### update filter weights
			if(step - MEM_SZ) % BATCH_SZ == 0:
				F1 += (dF1 + MOM_WEIGHT*dF1_mom)*EPS / BATCH_SZ
				F2 += (dF2 + MOM_WEIGHT*dF2_mom)*EPS / BATCH_SZ
				F3 += (dF3 + MOM_WEIGHT*dF3_mom)*EPS / BATCH_SZ
				FL += (dFL + MOM_WEIGHT*dFL_mom)*EPS / BATCH_SZ
				
				set_buffer(F1, F1_IND, filter_flag=1, gpu=GPU_CUR)
				set_buffer(F2, F2_IND, filter_flag=1, gpu=GPU_CUR)
				set_buffer(F3, F3_IND, filter_flag=1, gpu=GPU_CUR)
				set_buffer(FL, FL_IND, filter_flag=1, gpu=GPU_CUR)
				
				dF1_mom = copy.deepcopy(dF1)
				dF2_mom = copy.deepcopy(dF2)
				dF3_mom = copy.deepcopy(dF3)
				dFL_mom = copy.deepcopy(dFL)
				
				dF1 = np.zeros_like(dF1)
				dF2 = np.zeros_like(dF2)
				dF3 = np.zeros_like(dF3)
				dFL = np.zeros_like(dFL)
				
				network_updates += 1
				
				if network_updates % NETWORK_UPDATE == 0:
					print 'updating network'
					FL_prev = copy.deepcopy(FL)
					set_buffer(F1, F1_IND, filter_flag=1, gpu=GPU_PREV)
					set_buffer(F2, F2_IND, filter_flag=1, gpu=GPU_PREV)
					set_buffer(F3, F3_IND, filter_flag=1, gpu=GPU_PREV)
					set_buffer(FL, FL_IND, filter_flag=1, gpu=GPU_PREV)
					
		step += 1
		
		if step % SAVE_FREQ == 0:
			r_total_plot.append(r_total)
			err_plot.append(err)
			
			savemat(file_name + 'recent.mat', {'F1': F1, 'r_total_plot': r_total_plot, 'F2': F2, 'F3': F3, 'FL':FL, 'F1_init': F1_init, \
				'step': step, 'img': img, \
				'err_plot': err_plot, 'dogs_recent': dogs_recent, 'hearts_recent': hearts_recent, 'player_recent': player_recent, \
				'dog_show_flip_recent': dog_show_flip_recent, 'action_recent': action_recent, 'rotate_recent': rotate_recent, 'r_recent': r_recent})
			
			savemat(file_name + str(step) + '.mat', {'F1': F1, 'r_total_plot': r_total_plot, 'F2': F2, 'F3': F3, 'FL':FL, 'F1_init': F1_init, 'step': step, \
				'img': img, 'err_plot': err_plot})
			
			print 'step:', step, 'err:',err, 'r:',r_total, 'updates:',network_updates, 'eps:', CHANCE_RAND, 'F1:', np.max(F1), 't:',time.time() - t_start, file_name + str(step) + '.mat'
			
			err = 0
			r_total = 0
			
			t_start = time.time()
