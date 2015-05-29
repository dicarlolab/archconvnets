from archconvnets.unsupervised.cudnn_module.cudnn_module import *
import time
import numpy as np
from scipy.io import savemat, loadmat
import copy
from scipy.stats import zscore
import random
import scipy
from math import pi, sin, cos
from direct.showbase.ShowBase import ShowBase
from pandac.PandaModules import loadPrcFileData
import PIL
import PIL.Image

SAVE_FREQ = 2000
SAVE_CHECK_FREQ = 500000
EPS_GREED_FINAL = .1
EPS_GREED_FINAL_TIME = 4*5000000
GAMMA = 0.99
BATCH_SZ = 32
NETWORK_UPDATE = 10000
EPS = 1e-2
EPS_IMGNET = 1e-2
MOM_WEIGHT = 0.95
IMGNET_UPDATE_FREQ = 128 # run gradient descent once for this # of game steps

IMG_SZ = 92

N_MEAN_SAMPLES = 1000 # for mean image

F1_scale = 1e-3
F2_scale = 1e-3
F3_scale = 1e-3
FL_scale = 1e-4
FL_imgnet_scale = 1e-4

N = 64#32
n1 = N # L1 filters
n2 = N# ...
n3 = N

s3 = 3 # L1 filter size (px)
s2 = 4 # ...
s1 = 5

N_C = 3 # directions M, L, R

file_name = '/home/darren/reinforcement3d_saves/reinforcement_'

max_output_sz3  = 12

# these should all be different values:
GPU_CUR = 2
GPU_PREV = 3

# gpu buffer indices
MAX_OUTPUT1 = 0; DF2_DATA = 1; CONV_OUTPUT1 = 2; DPOOL1 = 3
F1_IND = 4; IMGS_PAD = 5; DF1 = 6; F2_IND = 11; 
D_UNPOOL2 = 12;F3_IND = 13; MAX_OUTPUT2 = 14; MAX_OUTPUT3 = 15
CONV_OUTPUT1 = 19; CONV_OUTPUT2 = 20; CONV_OUTPUT3 = 21
CONVA_OUTPUT1 = 50; CONVA_OUTPUT2 = 51; CONVA_OUTPUT3 = 52
DF2 = 25; DPOOL2 = 26; DF3_DATA = 27; DPOOL3 = 28; DF3 = 29; FL_PRED = 30;
FL_IND = 31; PRED = 32; DFL = 33; IMGS_PAD_IMGNET = 34

MAX_OUTPUT1_IMGNET = 35; CONV_OUTPUT1_IMGNET = 36
MAX_OUTPUT2_IMGNET = 37; MAX_OUTPUT3_IMGNET = 38
CONV_OUTPUT1_IMGNET = 39; CONV_OUTPUT2_IMGNET = 40; CONV_OUTPUT3_IMGNET = 41
IMGS_PAD_IMGNET_TEST = 42

MAX_OUTPUT1_IMGNET_TEST = 43; CONV_OUTPUT1_IMGNET_TEST = 44
MAX_OUTPUT2_IMGNET_TEST = 45; MAX_OUTPUT3_IMGNET_TEST = 46
CONV_OUTPUT1_IMGNET_TEST = 47; CONV_OUTPUT2_IMGNET_TEST = 48; CONV_OUTPUT3_IMGNET_TEST = 49

DA1 = 60; DA2 = 61; DA3 = 62

np.random.seed(6666)
F1 = np.single(np.random.normal(scale=F1_scale, size=(n1, 3, s1, s1)))
F2 = np.single(np.random.normal(scale=F2_scale, size=(n2, n1, s2, s2)))
F3 = np.single(np.random.normal(scale=F3_scale, size=(n3, n2, s3, s3)))
FL = np.single(np.random.normal(scale=FL_scale, size=(N_C, n1, max_output_sz3, max_output_sz3)))
FL_imgnet = np.single(np.random.normal(scale=FL_imgnet_scale, size=(999, n3, max_output_sz3, max_output_sz3)))

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
dFL_imgnet = np.zeros_like(FL_imgnet)

dF1_mom = np.zeros_like(F1)
dF2_mom = np.zeros_like(F2)
dF3_mom = np.zeros_like(F3)
dFL_mom = np.zeros_like(FL)
dFL_imgnet_mom = np.zeros_like(FL_imgnet)

r_total = 0
r_total_plot = []
network_updates = 0

step = 0
step_imgnet = 0

err = 0
err_imgnet = 0

err_plot = []
err_imgnet_plot = []
err_imgnet_test_plot = []
class_err_imgnet_test = []


########################################################################### init scene
loadPrcFileData("", "win-size " + str(IMG_SZ) + " " + str(IMG_SZ))
loadPrcFileData("", "window-type offscreen")
loadPrcFileData("", "require-window False")

PANDA_SZ = 2.5
KID_SZ = 1.25
ROOM_SZ = 9
ROOM_SZ_MV = 10
N_KIDS = 6
N_PANDAS = 6

ROT_RATE = 12
MOV_RATE = .75

app = ShowBase()

environ = app.loader.loadModel("/home/darren/panda_inst/share/panda3d/models/environment")
environ.reparentTo(app.render)
environ.setScale(0.25, 0.25, 0.25)
environ.setPos(-8, 42, 0)

h = 1

panda = []
for f in range(N_PANDAS):
	panda.append(app.loader.loadModel('/home/darren/panda_inst/share/panda3d/models/panda-model'))
	panda[f].setScale(.0015,.0015,.0025)
	panda[f].reparentTo(app.render)

kid = []
for f in range(N_KIDS):
	kid.append(app.loader.loadModel('/home/darren/models/ralph'))
	kid[f].setScale(0.25, 0.25, 0.25)
	kid[f].reparentTo(app.render)

################### random sample of initial conditions
def init_pos_vars():
	direction = 360*np.random.random()	
	kid_directions = 360*np.random.random(size=N_KIDS)
	panda_directions = 360*np.random.random(size=N_PANDAS)

	x = 2*ROOM_SZ*np.random.random() - ROOM_SZ
	y = 2*ROOM_SZ*np.random.random() - ROOM_SZ

	x_new = x; y_new = y

	collision = [1]
	while len(collision) != 0:
		panda_coords = 2*ROOM_SZ*np.random.random(size=(N_PANDAS,2)) - ROOM_SZ
		collision = np.nonzero(((x_new >= panda_coords[:,0]) * (x_new <= (panda_coords[:,0]+PANDA_SZ)) * (y_new >= panda_coords[:,1]) * (y_new <= (panda_coords[:,1]+PANDA_SZ))) + \
			((x_new+PANDA_SZ) >= panda_coords[:,0]) * ((x_new+PANDA_SZ) <= (panda_coords[:,0]+PANDA_SZ)) * (y_new >= panda_coords[:,1]) * (y_new <= (panda_coords[:,1]+PANDA_SZ)) + \
			((x_new+PANDA_SZ) >= panda_coords[:,0]) * ((x_new+PANDA_SZ) <= (panda_coords[:,0]+PANDA_SZ)) * ((y_new+PANDA_SZ) >= panda_coords[:,1]) * ((y_new+PANDA_SZ) <= (panda_coords[:,1]+PANDA_SZ)) + \
			(x_new >= panda_coords[:,0]) * (x_new <= (panda_coords[:,0]+PANDA_SZ)) * ((y_new+PANDA_SZ) >= panda_coords[:,1]) * ((y_new+PANDA_SZ) <= (panda_coords[:,1]+PANDA_SZ)))[0]

	collision = [1]
	while len(collision) != 0:
		kid_coords = 2*ROOM_SZ*np.random.random(size=(N_PANDAS,2)) - ROOM_SZ
		collision = np.nonzero(((x_new >= kid_coords[:,0]) * (x_new <= (kid_coords[:,0]+KID_SZ)) * (y_new >= kid_coords[:,1]) * (y_new <= (kid_coords[:,1]+KID_SZ))) + \
			((x_new+KID_SZ) >= kid_coords[:,0]) * ((x_new+KID_SZ) <= (kid_coords[:,0]+KID_SZ)) * (y_new >= kid_coords[:,1]) * (y_new <= (kid_coords[:,1]+KID_SZ)) + \
			((x_new+KID_SZ) >= kid_coords[:,0]) * ((x_new+KID_SZ) <= (kid_coords[:,0]+KID_SZ)) * ((y_new+KID_SZ) >= kid_coords[:,1]) * ((y_new+KID_SZ) <= (kid_coords[:,1]+KID_SZ)) + \
			(x_new >= kid_coords[:,0]) * (x_new <= (kid_coords[:,0]+KID_SZ)) * ((y_new+KID_SZ) >= kid_coords[:,1]) * ((y_new+KID_SZ) <= (kid_coords[:,1]+KID_SZ)))[0]
	
	return x,y, direction, kid_coords, panda_coords, kid_directions, panda_directions


############################################ render
def render(x,y, direction, panda, kid, kid_coords, panda_coords, kid_directions, panda_directions, filename='tmp2.png'):	
	app.camera.setPos(x,y,h)
	app.camera.setHpr(direction,0,0)
	
	for f in range(N_PANDAS):
		panda[f].setPos(panda_coords[f,0], panda_coords[f,1], 0)
		panda[f].setHpr(panda_directions[f],0,0)

	for f in range(N_KIDS):
		kid[f].setPos(kid_coords[f,0], kid_coords[f,1], 0)
		kid[f].setHpr(kid_directions[f],0,0)
		
	base.graphicsEngine.render_frame()
	base.screenshot(namePrefix=filename,defaultFilename = 0,source=app.win)
	
	return np.ascontiguousarray(np.asarray(PIL.Image.open(filename))[:,:,:3].transpose((2,0,1))[np.newaxis])

########################################## movement
def move(x,y, direction, kid_coords, panda_coords, kid_directions, panda_directions):
	dx = -MOV_RATE*sin(pi*direction/180)
	dy = MOV_RATE*cos(pi*direction/180)
	
	x_new = x + dx
	y_new = y + dy
	
	r = 0
	
	###################################### kid collision
	collision = np.nonzero(((x_new >= kid_coords[:,0]) * (x_new <= (kid_coords[:,0]+KID_SZ)) * (y_new >= kid_coords[:,1]) * (y_new <= (kid_coords[:,1]+KID_SZ))) + \
			((x_new+KID_SZ) >= kid_coords[:,0]) * ((x_new+KID_SZ) <= (kid_coords[:,0]+KID_SZ)) * (y_new >= kid_coords[:,1]) * (y_new <= (kid_coords[:,1]+KID_SZ)) + \
			((x_new+KID_SZ) >= kid_coords[:,0]) * ((x_new+KID_SZ) <= (kid_coords[:,0]+KID_SZ)) * ((y_new+KID_SZ) >= kid_coords[:,1]) * ((y_new+KID_SZ) <= (kid_coords[:,1]+KID_SZ)) + \
			(x_new >= kid_coords[:,0]) * (x_new <= (kid_coords[:,0]+KID_SZ)) * ((y_new+KID_SZ) >= kid_coords[:,1]) * ((y_new+KID_SZ) <= (kid_coords[:,1]+KID_SZ)))[0]
	
	r -= len(collision)
	
	while len(collision) != 0:
		kid_coords[collision] = 2*ROOM_SZ*np.random.random(size=(len(collision),2)) - ROOM_SZ
		kid_directions[collision] = 360*np.random.random(size=len(collision))
		
		collision = np.nonzero(((x_new >= kid_coords[:,0]) * (x_new <= (kid_coords[:,0]+KID_SZ)) * (y_new >= kid_coords[:,1]) * (y_new <= (kid_coords[:,1]+KID_SZ))) + \
			((x_new+KID_SZ) >= kid_coords[:,0]) * ((x_new+KID_SZ) <= (kid_coords[:,0]+KID_SZ)) * (y_new >= kid_coords[:,1]) * (y_new <= (kid_coords[:,1]+KID_SZ)) + \
			((x_new+KID_SZ) >= kid_coords[:,0]) * ((x_new+KID_SZ) <= (kid_coords[:,0]+KID_SZ)) * ((y_new+KID_SZ) >= kid_coords[:,1]) * ((y_new+KID_SZ) <= (kid_coords[:,1]+KID_SZ)) + \
			(x_new >= kid_coords[:,0]) * (x_new <= (kid_coords[:,0]+KID_SZ)) * ((y_new+KID_SZ) >= kid_coords[:,1]) * ((y_new+KID_SZ) <= (kid_coords[:,1]+KID_SZ)))[0]
	
	##################################### panda collision
	collision = np.nonzero(((x_new >= panda_coords[:,0]) * (x_new <= (panda_coords[:,0]+PANDA_SZ)) * (y_new >= panda_coords[:,1]) * (y_new <= (panda_coords[:,1]+PANDA_SZ))) + \
			((x_new+PANDA_SZ) >= panda_coords[:,0]) * ((x_new+PANDA_SZ) <= (panda_coords[:,0]+PANDA_SZ)) * (y_new >= panda_coords[:,1]) * (y_new <= (panda_coords[:,1]+PANDA_SZ)) + \
			((x_new+PANDA_SZ) >= panda_coords[:,0]) * ((x_new+PANDA_SZ) <= (panda_coords[:,0]+PANDA_SZ)) * ((y_new+PANDA_SZ) >= panda_coords[:,1]) * ((y_new+PANDA_SZ) <= (panda_coords[:,1]+PANDA_SZ)) + \
			(x_new >= panda_coords[:,0]) * (x_new <= (panda_coords[:,0]+PANDA_SZ)) * ((y_new+PANDA_SZ) >= panda_coords[:,1]) * ((y_new+PANDA_SZ) <= (panda_coords[:,1]+PANDA_SZ)))[0]

	r += len(collision)
	
	while len(collision) != 0:
		panda_coords[collision] = 2*ROOM_SZ*np.random.random(size=(len(collision),2)) - ROOM_SZ
		panda_directions[collision] = 360*np.random.random(size=len(collision))
		
		collision = np.nonzero(((x_new >= panda_coords[:,0]) * (x_new <= (panda_coords[:,0]+PANDA_SZ)) * (y_new >= panda_coords[:,1]) * (y_new <= (panda_coords[:,1]+PANDA_SZ))) + \
			((x_new+PANDA_SZ) >= panda_coords[:,0]) * ((x_new+PANDA_SZ) <= (panda_coords[:,0]+PANDA_SZ)) * (y_new >= panda_coords[:,1]) * (y_new <= (panda_coords[:,1]+PANDA_SZ)) + \
			((x_new+PANDA_SZ) >= panda_coords[:,0]) * ((x_new+PANDA_SZ) <= (panda_coords[:,0]+PANDA_SZ)) * ((y_new+PANDA_SZ) >= panda_coords[:,1]) * ((y_new+PANDA_SZ) <= (panda_coords[:,1]+PANDA_SZ)) + \
			(x_new >= panda_coords[:,0]) * (x_new <= (panda_coords[:,0]+PANDA_SZ)) * ((y_new+PANDA_SZ) >= panda_coords[:,1]) * ((y_new+PANDA_SZ) <= (panda_coords[:,1]+PANDA_SZ)))[0]
	
	if (x + dx) > -ROOM_SZ_MV and (x + dx) < ROOM_SZ_MV and\
		(y + dy) > -ROOM_SZ_MV and (y + dy) < ROOM_SZ_MV:
		x += dx
		y += dy
		
	return r, x,y, direction, kid_coords, panda_coords, kid_directions, panda_directions


#################################### mean image
mean_img = np.zeros((1,3, IMG_SZ,IMG_SZ), dtype='single')
for i in range(N_MEAN_SAMPLES):
	x,y, direction, kid_coords, panda_coords, kid_directions, panda_directions = init_pos_vars()
	mean_img += render(x,y, direction, panda, kid, kid_coords, panda_coords, kid_directions, panda_directions)
mean_img /= N_MEAN_SAMPLES

##################
# load test imgs into buffers (imgnet).. total: 4687 batches
N_BATCHES_IMGNET = 4687
N_BATCHES_IMGNET_TEST = 5
IMGNET_BATCH_SZ = 256
N_TRAIN_IMGNET = N_BATCHES_IMGNET_TEST*IMGNET_BATCH_SZ

imgnet_batch = N_BATCHES_IMGNET_TEST + 1

imgs_pad_test_imgnet = np.zeros((3, IMG_SZ, IMG_SZ, N_TRAIN_IMGNET),dtype='single')
Y_test_imgnet = np.zeros((N_TRAIN_IMGNET, 999),dtype='single')
labels_test_imgnet = np.zeros(N_TRAIN_IMGNET, dtype='int')

for batch in range(1,N_BATCHES_IMGNET_TEST+1):
	z2 = np.load('/export/storage/imgnet92/data_batch_' + str(batch))
	x2 = z2['data'].reshape((3, 92, 92, IMGNET_BATCH_SZ))

	labels_temp = np.asarray(z2['labels']).astype(int)
	
	Y_test_imgnet[(batch-1)*IMGNET_BATCH_SZ:batch*IMGNET_BATCH_SZ][range(IMGNET_BATCH_SZ), labels_temp] = 1
	labels_test_imgnet[(batch-1)*IMGNET_BATCH_SZ:batch*IMGNET_BATCH_SZ] = copy.deepcopy(labels_temp)

	imgs_pad_test_imgnet[:,:,:,(batch-1)*IMGNET_BATCH_SZ:batch*IMGNET_BATCH_SZ] = x2
imgs_pad_test_imgnet = np.ascontiguousarray(imgs_pad_test_imgnet.transpose((3,0,1,2)))
set_buffer(imgs_pad_test_imgnet - mean_img, IMGS_PAD_IMGNET_TEST, gpu=GPU_CUR)
Y_test_imgnet = Y_test_imgnet.T


##
panda_coords_recent = np.zeros((SAVE_FREQ, N_PANDAS, 2))
kid_coords_recent = np.zeros((SAVE_FREQ, N_KIDS, 2))

panda_directions_recent = np.zeros((SAVE_FREQ, N_PANDAS))
kid_directions_recent = np.zeros((SAVE_FREQ, N_KIDS))

x_recent = np.zeros(SAVE_FREQ)
y_recent = np.zeros(SAVE_FREQ)
direction_recent = np.zeros(SAVE_FREQ)

action_recent = np.zeros(SAVE_FREQ, dtype='int')
r_recent = np.zeros(SAVE_FREQ)

#########################################################################

t_start = time.time()

img = render(x,y, direction, panda, kid, kid_coords, panda_coords, kid_directions, panda_directions)

while True:
	# forward pass
	set_buffer(img - mean_img, IMGS_PAD, gpu=GPU_CUR)
	
	conv_buffers(F1_IND, IMGS_PAD, CONV_OUTPUT1, gpu=GPU_CUR)
	activation_buffers(CONV_OUTPUT1, CONVA_OUTPUT1, gpu=GPU_CUR)
	max_pool_cudnn_buffers(CONVA_OUTPUT1, MAX_OUTPUT1, gpu=GPU_CUR)
	
	conv_buffers(F2_IND, MAX_OUTPUT1, CONV_OUTPUT2, gpu=GPU_CUR)
	activation_buffers(CONV_OUTPUT2, CONVA_OUTPUT2, gpu=GPU_CUR)
	max_pool_cudnn_buffers(CONVA_OUTPUT2, MAX_OUTPUT2, gpu=GPU_CUR)
	
	conv_buffers(F3_IND, MAX_OUTPUT2, CONV_OUTPUT3, gpu=GPU_CUR)
	activation_buffers(CONV_OUTPUT3, CONVA_OUTPUT3, gpu=GPU_CUR)
	max_pool_cudnn_buffers(CONVA_OUTPUT3, MAX_OUTPUT3, gpu=GPU_CUR)
	
	max_output3 = return_buffer(MAX_OUTPUT3, gpu=GPU_CUR)
	
	pred = np.einsum(FL, range(4), max_output3, [4,1,2,3], [0])
	
	# choose action
	CHANCE_RAND = np.max((1 - ((1-EPS_GREED_FINAL)/EPS_GREED_FINAL_TIME)*step, EPS_GREED_FINAL))
	if np.random.rand() <= CHANCE_RAND:
		action = np.random.randint(3)
		if action >= 3: # increase likelihood of movement
			action = 0
	else:
		action = np.argmax(pred)
			
	# perform action
	r = 0
	if action == 0:
		r, x,y, direction, kid_coords, panda_coords, kid_directions, panda_directions = move(x,y, direction, kid_coords, panda_coords, kid_directions, panda_directions)
	elif action == 1:
		direction += ROT_RATE
	elif action == 2:
		direction -= ROT_RATE
	
	r_total += r
	
	##############
	# forward pass prev network on next frame
	img = render(x,y, direction, panda, kid, kid_coords, panda_coords, kid_directions, panda_directions)
	set_buffer(img - mean_img, IMGS_PAD, gpu=GPU_PREV)
	
	conv_buffers(F1_IND, IMGS_PAD, CONV_OUTPUT1, gpu=GPU_PREV)
	activation_buffers(CONV_OUTPUT1, CONV_OUTPUT1, gpu=GPU_PREV)
	max_pool_cudnn_buffers(CONV_OUTPUT1, MAX_OUTPUT1, gpu=GPU_PREV)
	
	conv_buffers(F2_IND, MAX_OUTPUT1, CONV_OUTPUT2, gpu=GPU_PREV)
	activation_buffers(CONV_OUTPUT2, CONV_OUTPUT2, gpu=GPU_PREV)
	max_pool_cudnn_buffers(CONV_OUTPUT2, MAX_OUTPUT2, gpu=GPU_PREV)
	
	conv_buffers(F3_IND, MAX_OUTPUT2, CONV_OUTPUT3, gpu=GPU_PREV)
	activation_buffers(CONV_OUTPUT3, CONV_OUTPUT3, gpu=GPU_PREV)
	max_pool_cudnn_buffers(CONV_OUTPUT3, MAX_OUTPUT3, gpu=GPU_PREV)

	# compute target
	max_output3 = return_buffer(MAX_OUTPUT3, gpu=GPU_PREV)
	pred_prev = np.einsum(FL_prev, range(4), max_output3, [4,1,2,3], [0])
	y_outputs = r + GAMMA * np.max(pred_prev)
	
	#############
	# gradients for current network
	pred_m_Y = y_outputs - pred[action]
	
	err += pred_m_Y**2
	
	FL_pred = np.ascontiguousarray((FL[action] * pred_m_Y)[np.newaxis])
	set_buffer(FL_pred, FL_PRED, gpu=GPU_CUR)
	
	########### backprop
	max_pool_back_cudnn_buffers(MAX_OUTPUT3, FL_PRED, CONVA_OUTPUT3, DPOOL3, gpu=GPU_CUR)
	activation_back_buffers(CONVA_OUTPUT3, DPOOL3, CONV_OUTPUT3, DA3, gpu=GPU_CUR)
	conv_dfilter_buffers(F3_IND, MAX_OUTPUT2, DA3, DF3, stream=3, gpu=GPU_CUR)
	conv_ddata_buffers(F3_IND, MAX_OUTPUT2, DA3, DF3_DATA, gpu=GPU_CUR)
	
	max_pool_back_cudnn_buffers(MAX_OUTPUT2, DF3_DATA, CONV_OUTPUT2, DPOOL2, gpu=GPU_CUR)
	activation_back_buffers(CONVA_OUTPUT2, DPOOL2, CONV_OUTPUT2, DA2, gpu=GPU_CUR)
	conv_dfilter_buffers(F2_IND, MAX_OUTPUT1, DA2, DF2, stream=2, gpu=GPU_CUR)
	conv_ddata_buffers(F2_IND, MAX_OUTPUT1, DA2, DF2_DATA, gpu=GPU_CUR)
	
	max_pool_back_cudnn_buffers(MAX_OUTPUT1, DF2_DATA, CONV_OUTPUT1, DPOOL1, gpu=GPU_CUR)
	activation_back_buffers(CONVA_OUTPUT1, DPOOL1, CONV_OUTPUT1, DA1, gpu=GPU_CUR)
	conv_dfilter_buffers(F1_IND, IMGS_PAD, DA1, DF1, stream=1, gpu=GPU_CUR)

	### return
	dFL[action] += max_output3[0]*pred_m_Y
	dF3 += return_buffer(DF3, stream=3, gpu=GPU_CUR)
	dF2 += return_buffer(DF2, stream=2, gpu=GPU_CUR)
	dF1 += return_buffer(DF1, stream=1, gpu=GPU_CUR)
	
	#### update filter weights
	if step % BATCH_SZ == 0:
		F1 += (dF1 + MOM_WEIGHT*dF1_mom)*EPS / BATCH_SZ
		F2 += (dF2 + MOM_WEIGHT*dF2_mom)*EPS / BATCH_SZ
		F3 += (dF3 + MOM_WEIGHT*dF3_mom)*EPS / BATCH_SZ
		FL += (dFL + MOM_WEIGHT*dFL_mom)*EPS / BATCH_SZ
		
		set_buffer(F1, F1_IND, filter_flag=1, gpu=GPU_CUR)
		set_buffer(F2, F2_IND, filter_flag=1, gpu=GPU_CUR)
		set_buffer(F3, F3_IND, filter_flag=1, gpu=GPU_CUR)
		
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
	
	#########################################################
	# imgnet learning
	if step % IMGNET_UPDATE_FREQ == 0:
		s_imgnet = step_imgnet % 4
		
		if s_imgnet == 0:
			##################
			# load train imgs into buffers
			z = np.load('/export/storage/imgnet92/data_batch_' + str(imgnet_batch))
			
			imgs_pad = np.zeros((IMGNET_BATCH_SZ, 3, IMG_SZ, IMG_SZ),dtype='single')
			Y_train = np.zeros((999, IMGNET_BATCH_SZ), dtype='uint8')

			x2 = z['data'].reshape((3, IMG_SZ, IMG_SZ, IMGNET_BATCH_SZ))

			labels = np.asarray(z['labels'])

			l = np.zeros((IMGNET_BATCH_SZ, 999),dtype='uint8')
			l[np.arange(IMGNET_BATCH_SZ),np.asarray(z['labels']).astype(int)] = 1
			Y_train = l.T

			imgs_pad = np.ascontiguousarray(x2.transpose((3,0,1,2)))
			
			imgnet_batch += 1
			if imgnet_batch > N_BATCHES_IMGNET:
				imgnet_batch = N_BATCHES_IMGNET_TEST + 1
		
		set_buffer(imgs_pad[s_imgnet*64:(s_imgnet+1)*64] - mean_img, IMGS_PAD_IMGNET, gpu=GPU_CUR)
		
		conv_buffers(F1_IND, IMGS_PAD_IMGNET, CONV_OUTPUT1_IMGNET, gpu=GPU_CUR)
		activation_buffers(CONV_OUTPUT1_IMGNET, CONV_OUTPUT1_IMGNET, gpu=GPU_CUR)
		max_pool_cudnn_buffers(CONV_OUTPUT1_IMGNET, MAX_OUTPUT1_IMGNET, gpu=GPU_CUR)
		
		conv_buffers(F2_IND, MAX_OUTPUT1_IMGNET, CONV_OUTPUT2_IMGNET, gpu=GPU_CUR)
		activation_buffers(CONV_OUTPUT2_IMGNET, CONV_OUTPUT2_IMGNET, gpu=GPU_CUR)
		max_pool_cudnn_buffers(CONV_OUTPUT2_IMGNET, MAX_OUTPUT2_IMGNET, gpu=GPU_CUR)
		
		conv_buffers(F3_IND, MAX_OUTPUT2_IMGNET, CONV_OUTPUT3_IMGNET, gpu=GPU_CUR)
		activation_buffers(CONV_OUTPUT2_IMGNET, CONV_OUTPUT2_IMGNET, gpu=GPU_CUR)
		max_pool_cudnn_buffers(CONV_OUTPUT3_IMGNET, MAX_OUTPUT3_IMGNET, gpu=GPU_CUR)
		
		max_output3 = return_buffer(MAX_OUTPUT3_IMGNET, gpu=GPU_CUR)
		
		pred = np.einsum(FL_imgnet, range(4), max_output3, [4,1,2,3], [0,4])
		pred_m_Y = Y_train[:,s_imgnet*64:(s_imgnet+1)*64] - pred
		
		err_imgnet += np.mean(pred_m_Y**2)
		
		dFL_imgnet = np.einsum(max_output3, range(4), pred_m_Y, [4,0], [4,1,2,3])
		FL_imgnet += (dFL_imgnet + MOM_WEIGHT*dFL_imgnet_mom)*EPS_IMGNET / 64		
		dFL_imgnet_mom = copy.deepcopy(dFL_imgnet)
		
		step_imgnet += 1
	
	################################3
	# debug/for visualizations
	save_loc = step % SAVE_FREQ
	
	panda_coords_recent[save_loc] = copy.deepcopy(panda_coords)
	kid_coords_recent[save_loc] = copy.deepcopy(kid_coords)

	panda_directions_recent[save_loc] = copy.deepcopy(panda_directions)
	kid_directions_recent[save_loc] = copy.deepcopy(kid_directions)

	x_recent[save_loc] = x
	y_recent[save_loc] = y
	direction_recent[save_loc] = direction

	action_recent[save_loc] = action
	r_recent[save_loc] = r
	
	step += 1
	
	if step % SAVE_FREQ == 0:
		###############################################
		# test imgs (imgnet); both action-based and control models
		conv_buffers(F1_IND, IMGS_PAD_IMGNET_TEST, CONV_OUTPUT1_IMGNET_TEST, gpu=GPU_CUR)
		activation_buffers(CONV_OUTPUT1_IMGNET_TEST, CONV_OUTPUT1_IMGNET_TEST, gpu=GPU_CUR)
		max_pool_cudnn_buffers(CONV_OUTPUT1_IMGNET_TEST, MAX_OUTPUT1_IMGNET_TEST, gpu=GPU_CUR)
		
		conv_buffers(F2_IND, MAX_OUTPUT1_IMGNET_TEST, CONV_OUTPUT2_IMGNET_TEST, gpu=GPU_CUR)
		activation_buffers(CONV_OUTPUT2_IMGNET_TEST, CONV_OUTPUT2_IMGNET_TEST, gpu=GPU_CUR)
		max_pool_cudnn_buffers(CONV_OUTPUT2_IMGNET_TEST, MAX_OUTPUT2_IMGNET_TEST, gpu=GPU_CUR)
		
		conv_buffers(F3_IND, MAX_OUTPUT2_IMGNET_TEST, CONV_OUTPUT3_IMGNET_TEST, gpu=GPU_CUR)
		activation_buffers(CONV_OUTPUT3_IMGNET_TEST, CONV_OUTPUT3_IMGNET_TEST, gpu=GPU_CUR)
		max_pool_cudnn_buffers(CONV_OUTPUT3_IMGNET_TEST, MAX_OUTPUT3_IMGNET_TEST, gpu=GPU_CUR)
		
		max_output3 = return_buffer(MAX_OUTPUT3_IMGNET_TEST, gpu=GPU_CUR)
		
		pred = np.einsum(FL_imgnet, range(4), max_output3, [4,1,2,3], [0,4])
			
		err_imgnet_test_plot.append(np.mean((pred - Y_test_imgnet)**2))
		class_err_imgnet_test.append(1-(np.argmax(pred,axis=0) == np.asarray(np.squeeze(labels_test_imgnet))).mean())
		
		##
		r_total_plot.append(r_total)
		err_plot.append(err)
		err_imgnet_plot.append(err_imgnet)
		
		savemat(file_name + 'recent.mat', {'F1': F1, 'r_total_plot': r_total_plot, 'F2': F2, 'F3': F3, 'FL':FL, 'F1_init': F1_init, 'FL_imgnet': FL_imgnet,\
			'step': step, 'img': img, 'err_plot': err_plot, 'SAVE_FREQ':SAVE_FREQ,\
			'SAVE_CHECK_FREQ':SAVE_CHECK_FREQ, 'EPS_GREED_FINAL':EPS_GREED_FINAL, 'EPS_GREED_FINAL_TIME':EPS_GREED_FINAL_TIME,\
			'GAMMA':GAMMA,'BATCH_SZ':BATCH_SZ,'NETWORK_UPDATE':NETWORK_UPDATE,'EPS':EPS,'EPS_IMGNET':EPS_IMGNET,'MOM_WEIGHT':MOM_WEIGHT,\
			'IMGNET_UPDATE_FREQ':IMGNET_UPDATE_FREQ,'N':N,\
			'panda_coords_recent': panda_coords_recent, 'kid_coords_recent': kid_coords_recent, \
			'panda_directions_recent': panda_directions_recent, 'kid_directions_recent': kid_directions_recent,
			'x_recent': x_recent, 'y_recent': y_recent, 'direction_recent': direction_recent, 'action_recent': action_recent,'err_imgnet_test_plot':err_imgnet_test_plot, 'class_err_imgnet_test':class_err_imgnet_test,\
			'r_recent': r_recent,'err_imgnet':err_imgnet})
			
		if step % SAVE_CHECK_FREQ == 0 :
			savemat(file_name + str(step) + '.mat', {'F1': F1, 'F2': F2, 'F3': F3, 'FL':FL, 'FL_imgnet':FL_imgnet,'F1_init': F1_init, 'step': step})
		
		print file_name + str(step) + '.mat'
		print 'step:', step, 'err:',err, 'r:',r_total, 'updates:',network_updates, 'eps:', CHANCE_RAND, 'F1:', np.max(F1), 't:',time.time() - t_start
		print 'err_imgnet:', err_imgnet, 'err_imgnet_test:',err_imgnet_test_plot[-1], \
			'class_imgnet_test:',class_err_imgnet_test[-1]
		
		err = 0
		err_imgnet = 0
		r_total = 0
		
		t_start = time.time()