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

load = True
file_name = '/export/imgnet_storage_full/reinforcement3d_more_objs_saves/reinforcement_'

#######################
# load/initialize variables
step_load = 0
if load == True:
	step_load = 4050000 #2250000
	z = loadmat(file_name + str(step_load) + '.mat')
	r_output = np.ascontiguousarray(np.squeeze(z['r_output']))
	y_outputs = np.ascontiguousarray(np.squeeze(z['y_outputs']))
	y_network_ver = np.ascontiguousarray(np.squeeze(z['y_network_ver']))
	x_output = np.ascontiguousarray(np.squeeze(z['x_output']))
	y_output = np.ascontiguousarray(np.squeeze(z['y_output']))
	direction_output = np.ascontiguousarray(np.squeeze(z['direction_output']))
	kid_directions_output = np.ascontiguousarray(np.squeeze(z['kid_directions_output']))
	kid_idents_output = np.ascontiguousarray(np.squeeze(z['kid_idents_output']))
	panda_directions_output = np.ascontiguousarray(np.squeeze(z['panda_directions_output']))
	panda_idents_output = np.ascontiguousarray(np.squeeze(z['panda_idents_output']))
	panda_coords_output = np.ascontiguousarray(np.squeeze(z['panda_coords_output']))
	kid_coords_output = np.ascontiguousarray(np.squeeze(z['kid_coords_output']))
	action_input = np.ascontiguousarray(np.squeeze(z['action_input']))
	direction_input = np.ascontiguousarray(np.squeeze(z['direction_input']))
	y_input = np.ascontiguousarray(np.squeeze(z['y_input']))
	x_input = np.ascontiguousarray(np.squeeze(z['x_input']))
	kid_directions_input = np.ascontiguousarray(np.squeeze(z['kid_directions_input']))
	kid_idents_input = np.ascontiguousarray(np.squeeze(z['kid_idents_input']))
	panda_directions_input = np.ascontiguousarray(np.squeeze(z['panda_directions_input']))
	panda_idents_input = np.ascontiguousarray(np.squeeze(z['panda_idents_input']))
	kid_coords_input = np.ascontiguousarray(np.squeeze(z['kid_coords_input']))
	panda_coords_input = np.ascontiguousarray(np.squeeze(z['panda_coords_input']))
	
	F1 = np.ascontiguousarray(z['F1'])
	F2 = np.ascontiguousarray(z['F2'])
	F3 = np.ascontiguousarray(z['F3'])
	FL = np.ascontiguousarray(z['FL'])
	FL_imgnet = np.ascontiguousarray(z['FL_imgnet'])
	FL_imgnet_scale = np.ascontiguousarray(z['FL_imgnet_scale'])
	F1_init = np.ascontiguousarray(z['F1_init'])
	step = np.int(z['step'])
	step_imgnet = np.int(z['step_imgnet'])
	MOV_RATE = float(z['MOV_RATE'])
	ROT_RATE = float(z['ROT_RATE'])
	N_PANDAS = np.int(z['N_PANDAS'])
	panda_coords_recent = np.ascontiguousarray(np.squeeze(z['panda_coords_recent']))
	kid_coords_recent = np.ascontiguousarray(np.squeeze(z['kid_coords_recent']))
	N_KIDS = np.int(z['N_KIDS'])
	ROOM_SZ_MV = float(z['ROOM_SZ_MV'])
	ROOM_SZ = np.int(z['ROOM_SZ'])
	panda_directions_recent = np.ascontiguousarray(np.squeeze(z['panda_directions_recent']))
	panda_idents_recent = np.ascontiguousarray(np.squeeze(z['panda_idents_recent']))
	kid_directions_recent = np.ascontiguousarray(np.squeeze(z['kid_directions_recent']))
	kid_idents_recent = np.ascontiguousarray(np.squeeze(z['kid_idents_recent']))
	KID_SZ = np.int(z['KID_SZ'])
	PANDA_SZ = np.int(z['PANDA_SZ'])
	max_output_sz3 = np.int(z['max_output_sz3'])
	x_recent = np.ascontiguousarray(np.squeeze(z['x_recent']))
	y_recent = np.ascontiguousarray(np.squeeze(z['y_recent']))
	direction_recent = np.ascontiguousarray(np.squeeze(z['direction_recent']))
	action_recent = np.ascontiguousarray(np.squeeze(z['action_recent']))

	#########
	
	r_recent = np.ascontiguousarray(np.squeeze(z['r_recent']))
	N = np.int(z['N'])
	F1_scale = np.single(z['F1_scale'])
	F2_scale = np.single(z['F2_scale'])
	F3_scale = np.single(z['F3_scale'])
	FL_scale = np.single(z['FL_scale'])
	IMGNET_UPDATE_FREQ = np.int(z['IMGNET_UPDATE_FREQ'])
	N_BATCHES_IMGNET = np.int(z['N_BATCHES_IMGNET'])
	N_BATCHES_IMGNET_TEST = np.int(z['N_BATCHES_IMGNET_TEST'])
	IMGNET_BATCH_SZ = np.int(z['IMGNET_BATCH_SZ'])
	N_MEAN_SAMPLES = np.int(z['N_MEAN_SAMPLES'])
	IMG_SZ = np.int(z['IMG_SZ'])
	MOM_WEIGHT = np.single(z['MOM_WEIGHT'])
	EPS = np.single(z['EPS'])
	NETWORK_UPDATE = np.int(z['NETWORK_UPDATE'])
	BATCH_SZ = np.int(z['BATCH_SZ'])
	GAMMA = float(z['GAMMA'])
	EPS_GREED_FINAL_TIME = float(z['EPS_GREED_FINAL_TIME'])
	EPS_GREED_FINAL = float(z['EPS_GREED_FINAL'])
	EPS_IMGNET = float(z['EPS_IMGNET'])
	MEM_SZ = np.int(z['MEM_SZ'])
	SAVE_CHECK_FREQ = np.int(z['SAVE_CHECK_FREQ'])
	SAVE_FREQ = np.int(z['SAVE_FREQ'])
	r_total_plot = np.squeeze(z['r_total_plot']).tolist()
	err_plot = np.squeeze(z['err_plot']).tolist()
	err_imgnet_test_plot = np.squeeze(z['err_imgnet_test_plot']).tolist()
	err_imgnet_plot = np.squeeze(z['err_imgnet_plot']).tolist()
	class_err_imgnet_test = np.squeeze(z['class_err_imgnet_test']).tolist()
	network_updates = step / NETWORK_UPDATE #np.int(z['network_updates'])
	FL_prev = copy.deepcopy(FL)
else:
	SAVE_FREQ = 2000
	MEM_SZ = 1000000
	SAVE_CHECK_FREQ = 150000
	EPS_GREED_FINAL = .1
	EPS_GREED_FINAL_TIME = 5000000
	GAMMA = 0.99
	BATCH_SZ = 32
	NETWORK_UPDATE = 10000
	EPS = 2.5e-3
	EPS_IMGNET = 5e-4
	MOM_WEIGHT = 0.95
	IMGNET_UPDATE_FREQ = 128*2 # run gradient descent once for this # of game steps

	IMG_SZ = 92

	N_MEAN_SAMPLES = 1000 # for mean image

	F1_scale = 1e-2
	F2_scale = 1e-2
	F3_scale = 1e-2
	FL_scale = 1e-2
	FL_imgnet_scale = 1e-3

	N = 64#32
	n1 = N # L1 filters
	n2 = N# ...
	n3 = N

	s3 = 3 # L1 filter size (px)
	s2 = 4 # ...
	s1 = 5

	max_output_sz3  = 12
	
	np.random.seed(6666)
	F1 = np.single(np.random.normal(scale=F1_scale, size=(n1, 3, s1, s1)))
	F2 = np.single(np.random.normal(scale=F2_scale, size=(n2, n1, s2, s2)))
	F3 = np.single(np.random.normal(scale=F3_scale, size=(n3, n2, s3, s3)))
	FL = np.single(np.random.normal(scale=FL_scale, size=(3, n1, max_output_sz3, max_output_sz3)))
	FL_imgnet = np.single(np.random.normal(scale=FL_imgnet_scale, size=(999, n3, max_output_sz3, max_output_sz3)))
	
	r_total = 0
	r_total_plot = []
	network_updates = 0

	step = 0
	step_imgnet = 0

	err_plot = []
	err_imgnet_plot = []
	err_imgnet_test_plot = []
	class_err_imgnet_test = []

	# panda/world variables
	PANDA_SZ = 1.25
	KID_SZ = 1.25
	ROOM_SZ = 9
	ROOM_SZ_MV = 10
	N_KIDS = 6
	N_PANDAS = 6

	ROT_RATE = 12
	MOV_RATE = .75

	# imgnet eval params
	N_BATCHES_IMGNET = 4687
	N_BATCHES_IMGNET_TEST = 5
	IMGNET_BATCH_SZ = 256
	
	# recent buffers for visualization/debugging
	panda_coords_recent = np.zeros((SAVE_FREQ, N_PANDAS, 2))
	kid_coords_recent = np.zeros((SAVE_FREQ, N_KIDS, 2))

	panda_directions_recent = np.zeros((SAVE_FREQ, N_PANDAS))
	kid_directions_recent = np.zeros((SAVE_FREQ, N_KIDS))
	
	panda_idents_recent = np.zeros((SAVE_FREQ, N_PANDAS),dtype='int')
	kid_idents_recent = np.zeros((SAVE_FREQ, N_KIDS),dtype='int')

	x_recent = np.zeros(SAVE_FREQ)
	y_recent = np.zeros(SAVE_FREQ)
	direction_recent = np.zeros(SAVE_FREQ)

	action_recent = np.zeros(SAVE_FREQ, dtype='int')
	r_recent = np.zeros(SAVE_FREQ)
	
	# replay buffers
	panda_coords_input = np.zeros((MEM_SZ, N_PANDAS, 2))
	kid_coords_input = np.zeros((MEM_SZ, N_KIDS, 2))

	panda_directions_input = np.zeros((MEM_SZ, N_PANDAS))
	kid_directions_input = np.zeros((MEM_SZ, N_KIDS))
	
	panda_idents_input = np.zeros((MEM_SZ, N_PANDAS),dtype='int')
	kid_idents_input = np.zeros((MEM_SZ, N_KIDS),dtype='int')

	x_input = np.zeros(MEM_SZ)
	y_input = np.zeros(MEM_SZ)
	direction_input = np.zeros(MEM_SZ)

	action_input = np.zeros(MEM_SZ, dtype='int')
	
	##
	panda_coords_output = np.zeros((MEM_SZ, N_PANDAS, 2))
	kid_coords_output = np.zeros((MEM_SZ, N_KIDS, 2))

	panda_directions_output = np.zeros((MEM_SZ, N_PANDAS))
	kid_directions_output = np.zeros((MEM_SZ, N_KIDS))
	
	panda_idents_output = np.zeros((MEM_SZ, N_PANDAS),dtype='int')
	kid_idents_output = np.zeros((MEM_SZ, N_KIDS),dtype='int')

	x_output = np.zeros(MEM_SZ)
	y_output = np.zeros(MEM_SZ)
	direction_output = np.zeros(MEM_SZ)

	##
	r_output = np.zeros(MEM_SZ)
	y_outputs = np.zeros(MEM_SZ)
	y_network_ver = -np.ones(MEM_SZ)

# these should all be different values:
GPU_CUR = 0
GPU_PREV = 1

animate = ['foreign_cat','jaguar','longhair_cat','tiger','lion','semilonghair_cat','shorthair_cat',
	'leopard','oriental','coyote','pug','doberman','dalmatian','schnauzer',
	'bloodhound','bullmastiff','chihuahua','goldenretriever','weimaraner',
	'fieldmouse','hare', 'MB30418','crocodile', 'elephant','goat',	'elk', 
	'lion','lo_poly_animal_TRTL_B','lo_poly_animal_ELE_AS1', 'lo_poly_animal_TRANTULA',
	'lo_poly_animal_HRS_ARBN','lo_poly_animal_CHICKDEE',   'lo_poly_animal_PENGUIN',
	'Air_hostess_pose01','Engineer_pose01', 'Fireman_pose01',  'Medic_pose11',
	'Professor_pose04','Soldier_pose01',  'Soldier_pose08','Workman_pose01',   'Workman_pose10']

inanimate = ['001M', '002M', '003M', '004M','082M', '087M', '093M', '051M','076M',
	 '054M','07_guitar', '17_el_guitar', '15_violin', '21_violin', 'single_melon','single_cucumber']

tmp_filename = 'tmp' + str(GPU_CUR) + str(GPU_PREV) + '2.png' # panda
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
err = 0
err_imgnet = 0

########################################################################### init scene
loadPrcFileData("", "win-size " + str(IMG_SZ) + " " + str(IMG_SZ))
loadPrcFileData("", "window-type offscreen")
loadPrcFileData("", "require-window False")

app = ShowBase()

environ = app.loader.loadModel("/home/darren/panda_inst/share/panda3d/models/environment")
environ.reparentTo(app.render)
environ.setScale(0.25, 0.25, 0.25)
environ.setPos(-8, 42, 0)

h = 1

print 'loading models'

#### canonical model scaling/orientation
rotations_animate = np.zeros(len(animate))
model_scales_animate = 3*np.ones(len(animate))
rotate_animate = np.zeros(len(animate))
offset_animate = np.zeros(len(animate))

model_scales_animate[6:9] = 2
offset_animate[6:9] = .5

model_scales_animate[11:16] = 2
offset_animate[11:16] = .5

model_scales_animate[19] = 22

model_scales_animate[20] = 8

model_scales_animate[23] = 1

model_scales_animate[25] = 1.5

offset_animate[21] = .2
offset_animate[22] = .2

offset_animate[27] = .3
model_scales_animate[27] = 10
rotate_animate[27:29] = -90

model_scales_animate[28] = .45

model_scales_animate[29] = 10
rotate_animate[29] = -90

model_scales_animate[33-3] = .8
rotate_animate[33-3:36-3] = -90

model_scales_animate[34-3] = 24

model_scales_animate[36-3:45-3] = .025
rotate_animate[36-3:45-3] = -90
offset_animate[36-3:45-3] = 1

rotations_inanimate = np.zeros(len(inanimate))
model_scales_inanimate = 2*np.ones(len(inanimate))
rotate_inanimate = np.zeros(len(inanimate))
offset_inanimate = np.zeros(len(inanimate))

model_scales_inanimate[10:14] = .01
rotate_inanimate[10:14] = -90
offset_inanimate[10:14] = 1

model_scales_inanimate[14] = .25
rotate_inanimate[14] = -90

model_scales_inanimate[15] = .2
rotate_inanimate[15] = -90
offset_inanimate[15] = .1

model_scales_inanimate[11] = .002

model_scales_inanimate[10] = .014
offset_inanimate[10] = 1

animate_models = []
for ind in range(len(animate)):
	animate_models.append(app.loader.loadModel('/home/darren/.skdata/genthor/resources/eggs/' + animate[ind] + '/' + animate[ind] + '.egg'))
	animate_models[ind].reparentTo(app.render)
	animate_models[ind].setPos(-50.6, 70.5, -10)
	animate_models[ind].setScale(model_scales_animate[ind], model_scales_animate[ind], model_scales_animate[ind])

inanimate_models = []
for ind in range(len(inanimate)):
	inanimate_models.append(app.loader.loadModel('/home/darren/.skdata/genthor/resources/eggs/' + inanimate[ind] + '/' + inanimate[ind] + '.egg'))
	inanimate_models[ind].reparentTo(app.render)
	inanimate_models[ind].setPos(-50.6, 70.5, -10)
	inanimate_models[ind].setScale(model_scales_inanimate[ind], model_scales_inanimate[ind], model_scales_inanimate[ind])


print 'done'

################## random sample of initial conditions
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
	
	panda_idents = np.arange(len(inanimate))
	kid_idents = np.arange(len(animate))
	random.shuffle(panda_idents)
	random.shuffle(kid_idents)
	panda_idents = panda_idents[:N_PANDAS]
	kid_idents = kid_idents[:N_KIDS]
	
	return x,y, direction, kid_coords, panda_coords, kid_directions, panda_directions, panda_idents, kid_idents



############################################ render
def render(x,y, direction, animate_models, inanimate_models, kid_coords, panda_coords, kid_directions, panda_directions, panda_idents, kid_idents, filename=tmp_filename):	
	app.camera.setPos(x,y,h)
	app.camera.setHpr(direction,0,0)
	
	# reset poses
	for f in range(len(inanimate)):
		inanimate_models[f].setPos(0, 0, -10)
		
	for f in range(len(animate)):
		animate_models[f].setPos(0, 0, -10)
	
	# of the shown models, update their poses
	for f in range(N_PANDAS):
		inanimate_models[panda_idents[f]].setPos(panda_coords[f,0], panda_coords[f,1], offset_inanimate[panda_idents[f]])
		inanimate_models[panda_idents[f]].setHpr(panda_directions[f],rotate_inanimate[panda_idents[f]],0)

	for f in range(N_KIDS):
		animate_models[kid_idents[f]].setPos(kid_coords[f,0], kid_coords[f,1], offset_animate[kid_idents[f]])
		animate_models[kid_idents[f]].setHpr(kid_directions[f],rotate_animate[kid_idents[f]],0)
		
	base.graphicsEngine.render_frame()
	base.screenshot(namePrefix=filename,defaultFilename = 0,source=app.win)
	
	return np.ascontiguousarray(np.asarray(PIL.Image.open(filename))[:,:,:3].transpose((2,0,1))[np.newaxis])

########################################## movement
def move(x,y, direction, kid_coords, panda_coords, kid_directions, panda_directions, panda_idents, kid_idents):
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
	
	collision2 = copy.deepcopy(collision)
	
	while (len(collision2) != 0) or (len(np.unique(kid_idents)) != N_KIDS):
		kid_coords[collision] = 2*ROOM_SZ*np.random.random(size=(len(collision),2)) - ROOM_SZ
		kid_directions[collision] = 360*np.random.random(size=len(collision))
		kid_idents[collision] = np.random.randint(len(animate), size=len(collision))
		
		collision2 = np.nonzero(((x_new >= kid_coords[:,0]) * (x_new <= (kid_coords[:,0]+KID_SZ)) * (y_new >= kid_coords[:,1]) * (y_new <= (kid_coords[:,1]+KID_SZ))) + \
			((x_new+KID_SZ) >= kid_coords[:,0]) * ((x_new+KID_SZ) <= (kid_coords[:,0]+KID_SZ)) * (y_new >= kid_coords[:,1]) * (y_new <= (kid_coords[:,1]+KID_SZ)) + \
			((x_new+KID_SZ) >= kid_coords[:,0]) * ((x_new+KID_SZ) <= (kid_coords[:,0]+KID_SZ)) * ((y_new+KID_SZ) >= kid_coords[:,1]) * ((y_new+KID_SZ) <= (kid_coords[:,1]+KID_SZ)) + \
			(x_new >= kid_coords[:,0]) * (x_new <= (kid_coords[:,0]+KID_SZ)) * ((y_new+KID_SZ) >= kid_coords[:,1]) * ((y_new+KID_SZ) <= (kid_coords[:,1]+KID_SZ)))[0]
	
	##################################### panda collision
	collision = np.nonzero(((x_new >= panda_coords[:,0]) * (x_new <= (panda_coords[:,0]+PANDA_SZ)) * (y_new >= panda_coords[:,1]) * (y_new <= (panda_coords[:,1]+PANDA_SZ))) + \
			((x_new+PANDA_SZ) >= panda_coords[:,0]) * ((x_new+PANDA_SZ) <= (panda_coords[:,0]+PANDA_SZ)) * (y_new >= panda_coords[:,1]) * (y_new <= (panda_coords[:,1]+PANDA_SZ)) + \
			((x_new+PANDA_SZ) >= panda_coords[:,0]) * ((x_new+PANDA_SZ) <= (panda_coords[:,0]+PANDA_SZ)) * ((y_new+PANDA_SZ) >= panda_coords[:,1]) * ((y_new+PANDA_SZ) <= (panda_coords[:,1]+PANDA_SZ)) + \
			(x_new >= panda_coords[:,0]) * (x_new <= (panda_coords[:,0]+PANDA_SZ)) * ((y_new+PANDA_SZ) >= panda_coords[:,1]) * ((y_new+PANDA_SZ) <= (panda_coords[:,1]+PANDA_SZ)))[0]

	r += len(collision)
	
	collision2 = copy.deepcopy(collision)
	
	while (len(collision2) != 0) or (len(np.unique(panda_idents)) != N_PANDAS):
		panda_coords[collision] = 2*ROOM_SZ*np.random.random(size=(len(collision),2)) - ROOM_SZ
		panda_directions[collision] = 360*np.random.random(size=len(collision))
		panda_idents[collision] = np.random.randint(len(inanimate), size=len(collision))
		
		collision2 = np.nonzero(((x_new >= panda_coords[:,0]) * (x_new <= (panda_coords[:,0]+PANDA_SZ)) * (y_new >= panda_coords[:,1]) * (y_new <= (panda_coords[:,1]+PANDA_SZ))) + \
			((x_new+PANDA_SZ) >= panda_coords[:,0]) * ((x_new+PANDA_SZ) <= (panda_coords[:,0]+PANDA_SZ)) * (y_new >= panda_coords[:,1]) * (y_new <= (panda_coords[:,1]+PANDA_SZ)) + \
			((x_new+PANDA_SZ) >= panda_coords[:,0]) * ((x_new+PANDA_SZ) <= (panda_coords[:,0]+PANDA_SZ)) * ((y_new+PANDA_SZ) >= panda_coords[:,1]) * ((y_new+PANDA_SZ) <= (panda_coords[:,1]+PANDA_SZ)) + \
			(x_new >= panda_coords[:,0]) * (x_new <= (panda_coords[:,0]+PANDA_SZ)) * ((y_new+PANDA_SZ) >= panda_coords[:,1]) * ((y_new+PANDA_SZ) <= (panda_coords[:,1]+PANDA_SZ)))[0]
	
	if (x + dx) > -ROOM_SZ_MV and (x + dx) < ROOM_SZ_MV and\
		(y + dy) > -ROOM_SZ_MV and (y + dy) < ROOM_SZ_MV:
		x += dx
		y += dy
		
	return r, x,y, direction, kid_coords, panda_coords, kid_directions, panda_directions, panda_idents, kid_idents


#################################### mean image
mean_img = np.zeros((1,3, IMG_SZ,IMG_SZ), dtype='single')
for i in range(N_MEAN_SAMPLES):
	x,y, direction, kid_coords, panda_coords, kid_directions, panda_directions, panda_idents, kid_idents = init_pos_vars()
	mean_img += render(x,y, direction, animate_models, inanimate_models, kid_coords, panda_coords, kid_directions, panda_directions, panda_idents, kid_idents)
mean_img /= N_MEAN_SAMPLES

##################
# load test imgs into buffers (imgnet).. total: 4687 batches
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
imgnet_loaded = False

#########################################################################
print 'starting'
t_start = time.time()

while True:
	mem_loc  = step % MEM_SZ
	
	# copy current state
	panda_coords_input[mem_loc] = copy.deepcopy(panda_coords)
	kid_coords_input[mem_loc] = copy.deepcopy(kid_coords)

	panda_directions_input[mem_loc] = copy.deepcopy(panda_directions)
	kid_directions_input[mem_loc] = copy.deepcopy(kid_directions)
	
	panda_idents_input[mem_loc] = copy.deepcopy(panda_idents)
	kid_idents_input[mem_loc] = copy.deepcopy(kid_idents)

	x_input[mem_loc] = x
	y_input[mem_loc] = y
	direction_input[mem_loc] = direction
	
	# choose action
	network_outputs_computed = False
	CHANCE_RAND = np.max((1 - ((1-EPS_GREED_FINAL)/EPS_GREED_FINAL_TIME)*(step - MEM_SZ), EPS_GREED_FINAL))
	if np.random.rand() <= CHANCE_RAND:
		action = np.random.randint(3)
		if action >= 3: # increase likelihood of movement
			action = 0
	else:
		img = render(x,y, direction, animate_models, inanimate_models, kid_coords, panda_coords, kid_directions, panda_directions, panda_idents, kid_idents)
		network_outputs_computed = True
		
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
		
		action = np.argmax(pred)
			
	# perform action
	r = 0
	if action == 0:
		r, x,y, direction, kid_coords, panda_coords, kid_directions, panda_directions, panda_idents, kid_idents = move(x,y, direction, kid_coords, panda_coords, kid_directions, panda_directions, panda_idents, kid_idents)
	elif action == 1:
		direction += ROT_RATE
	elif action == 2:
		direction -= ROT_RATE
	
	r_total += r
	
	# copy current state
	panda_coords_output[mem_loc] = copy.deepcopy(panda_coords)
	kid_coords_output[mem_loc] = copy.deepcopy(kid_coords)

	panda_directions_output[mem_loc] = copy.deepcopy(panda_directions)
	kid_directions_output[mem_loc] = copy.deepcopy(kid_directions)
	
	panda_idents_output[mem_loc] = copy.deepcopy(panda_idents)
	kid_idents_output[mem_loc] = copy.deepcopy(kid_idents)

	x_output[mem_loc] = x
	y_output[mem_loc] = y
	direction_output[mem_loc] = direction
	
	r_output[mem_loc] = r
	action_input[mem_loc] = action
	
	if network_outputs_computed == True: # otherwise, the input wasn't actually fed through
		y_outputs[mem_loc] = r + GAMMA * np.max(pred)
		y_network_ver[mem_loc] = network_updates % NETWORK_UPDATE
	else:
		y_network_ver[mem_loc] = -1
	
	# debug/for visualizations
	save_loc = step % SAVE_FREQ
	
	panda_coords_recent[save_loc] = copy.deepcopy(panda_coords)
	kid_coords_recent[save_loc] = copy.deepcopy(kid_coords)

	panda_directions_recent[save_loc] = copy.deepcopy(panda_directions)
	kid_directions_recent[save_loc] = copy.deepcopy(kid_directions)
	
	panda_idents_recent[save_loc] = copy.deepcopy(panda_idents)
	kid_idents_recent[save_loc] = copy.deepcopy(kid_idents)

	x_recent[save_loc] = x
	y_recent[save_loc] = y
	direction_recent[save_loc] = direction

	action_recent[save_loc] = action
	r_recent[save_loc] = r

	if step == MEM_SZ:
		print 'beginning gradient computations'
	
	######################################
	# update gradient?
	if step >= MEM_SZ:
		trans = np.random.randint(MEM_SZ)
		#trans = mem_loc
	
		img_cur = render(x_input[trans],y_input[trans], direction_input[trans], animate_models, inanimate_models, kid_coords_input[trans], \
			panda_coords_input[trans], kid_directions_input[trans], panda_directions_input[trans],\
			panda_idents_input[trans], kid_idents_input[trans])
		
		set_buffer(img_cur - mean_img, IMGS_PAD, gpu=GPU_CUR)
		
		################################################### gradients for the action-based model
		# forward pass current network
		conv_buffers(F1_IND, IMGS_PAD, CONV_OUTPUT1, gpu=GPU_CUR)
		activation_buffers(CONV_OUTPUT1, CONVA_OUTPUT1, gpu=GPU_CUR)
		max_pool_cudnn_buffers(CONVA_OUTPUT1, MAX_OUTPUT1, gpu=GPU_CUR)
		
		conv_buffers(F2_IND, MAX_OUTPUT1, CONV_OUTPUT2, gpu=GPU_CUR)
		activation_buffers(CONV_OUTPUT2, CONVA_OUTPUT2, gpu=GPU_CUR)
		max_pool_cudnn_buffers(CONVA_OUTPUT2, MAX_OUTPUT2, gpu=GPU_CUR)
		
		conv_buffers(F3_IND, MAX_OUTPUT2, CONV_OUTPUT3, gpu=GPU_CUR)
		activation_buffers(CONV_OUTPUT3, CONVA_OUTPUT3, gpu=GPU_CUR)
		max_pool_cudnn_buffers(CONVA_OUTPUT3, MAX_OUTPUT3, gpu=GPU_CUR)
			
		# forward pass prev network
		# (only compute if we have not already computed the output for this version of the network)
		if y_network_ver[trans] != (network_updates % NETWORK_UPDATE):
			img_prev = render(x_output[trans],y_output[trans], direction_output[trans], animate_models, inanimate_models, kid_coords_output[trans], \
				panda_coords_output[trans], kid_directions_output[trans], panda_directions_output[trans],\
				panda_idents_output[trans], kid_idents_output[trans])
			set_buffer(img_prev - mean_img, IMGS_PAD, gpu=GPU_PREV)
			
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
			y_outputs[trans] = r_output[trans] + GAMMA * np.max(pred_prev)
			y_network_ver[trans] = network_updates % NETWORK_UPDATE
			
		max_output3 = return_buffer(MAX_OUTPUT3, gpu=GPU_CUR)
		
		pred = np.einsum(FL, range(4), max_output3, [4,1,2,3], [0])
		pred_m_Y = y_outputs[trans] - pred[action_input[trans]]
		
		err += pred_m_Y**2
		
		FL_pred = np.ascontiguousarray((FL[action_input[trans]] * pred_m_Y)[np.newaxis])
		
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
		dFL[action_input[trans]] += max_output3[0]*pred_m_Y
		dF3 += return_buffer(DF3, stream=3, gpu=GPU_CUR)
		dF2 += return_buffer(DF2, stream=2, gpu=GPU_CUR)
		dF1 += return_buffer(DF1, stream=1, gpu=GPU_CUR)
		
		#########################################################
		# imgnet learning: both action model and control model gradients
		if step % IMGNET_UPDATE_FREQ == 0:
			s_imgnet = step_imgnet % 2
			
			if s_imgnet == 0 or imgnet_loaded == False:
				imgnet_loaded = True
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
			
			set_buffer(imgs_pad[s_imgnet*128:(s_imgnet+1)*128] - mean_img, IMGS_PAD_IMGNET, gpu=GPU_CUR)
			
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
			
			pred_m_Y = Y_train[:,s_imgnet*128:(s_imgnet+1)*128] - pred
			
			err_imgnet += np.mean(pred_m_Y**2)
			
			dFL_imgnet = np.einsum(max_output3, range(4), pred_m_Y, [4,0], [4,1,2,3])
			
			FL_imgnet += (dFL_imgnet + MOM_WEIGHT*dFL_imgnet_mom)*EPS_IMGNET / 128
			
			dFL_imgnet_mom = copy.deepcopy(dFL_imgnet)
			
			step_imgnet += 1
		
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
	
	if step % SAVE_FREQ == 0:#and step >= MEM_SZ:
		if step >= MEM_SZ:
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
		
		img = render(x,y, direction, animate_models, inanimate_models, kid_coords, panda_coords, kid_directions, panda_directions, panda_idents, kid_idents)
		
		dic = {'F1': F1, 'r_total_plot': r_total_plot, 'F2': F2, 'F3': F3, 'FL':FL, \
			'F1_init': F1_init, 'err_imgnet_plot':err_imgnet_plot,\
			'FL_imgnet': FL_imgnet,'KID_SZ':KID_SZ, 'PANDA_SZ':PANDA_SZ,\
			'step': step, 'img': img, 'err_plot': err_plot, 'SAVE_FREQ':SAVE_FREQ,\
			'F1_scale':F1_scale,'F2_scale':F2_scale,'F3_scale':F3_scale,\
			'FL_scale':FL_scale, 'FL_imgnet_scale':FL_imgnet_scale, 'N_MEAN_SAMPLES':N_MEAN_SAMPLES,\
			'SAVE_CHECK_FREQ':SAVE_CHECK_FREQ, 'EPS_GREED_FINAL':EPS_GREED_FINAL,\
			'EPS_GREED_FINAL_TIME':EPS_GREED_FINAL_TIME,'N_KIDS':N_KIDS, \
			'ROOM_SZ_MV':ROOM_SZ_MV, 'ROOM_SZ':ROOM_SZ,'step_imgnet':step_imgnet,\
			'GAMMA':GAMMA,'BATCH_SZ':BATCH_SZ,'NETWORK_UPDATE':NETWORK_UPDATE,'EPS':EPS,'EPS_IMGNET':EPS_IMGNET,'MOM_WEIGHT':MOM_WEIGHT,'MOV_RATE':MOV_RATE,'ROT_RATE':ROT_RATE, 'N_PANDAS':N_PANDAS,\
			'IMGNET_UPDATE_FREQ':IMGNET_UPDATE_FREQ,'N':N,'max_output_sz3':max_output_sz3,\
			'panda_coords_recent': panda_coords_recent, \
			'kid_coords_recent': kid_coords_recent, \
			'panda_directions_recent': panda_directions_recent,'panda_idents_recent': panda_idents_recent, \
			'kid_directions_recent': kid_directions_recent,'kid_idents_recent': kid_idents_recent,\
			'x_recent': x_recent, 'y_recent': y_recent, 'direction_recent': direction_recent, \
			'action_recent': action_recent,'err_imgnet_test_plot':err_imgnet_test_plot,\
			'class_err_imgnet_test':class_err_imgnet_test,'IMG_SZ':IMG_SZ,\
			'r_recent': r_recent,'err_imgnet':err_imgnet,'network_updates':network_updates,\
			'N_BATCHES_IMGNET':N_BATCHES_IMGNET,'MEM_SZ':MEM_SZ,\
			'N_BATCHES_IMGNET_TEST':N_BATCHES_IMGNET_TEST,\
			'IMGNET_BATCH_SZ':IMGNET_BATCH_SZ,'SAVE_CHECK_FREQ':SAVE_CHECK_FREQ}
		
		savemat(file_name + 'recent.mat', dic)
		
		if step % SAVE_CHECK_FREQ == 0 :
			print 'saving buffers...'
			######### buffers for replay
			dic.update({'panda_coords_input': panda_coords_input, \
				'kid_coords_input':kid_coords_input,\
				'panda_directions_input':panda_directions_input, \
				'panda_idents_input':panda_idents_input, \
				'kid_directions_input':kid_directions_input,\
				'kid_idents_input':kid_idents_input,\
				'x_input':x_input,'y_input':y_input,'direction_input':direction_input,\
				'action_input':action_input,\
				'panda_coords_output':panda_coords_output,\
				'kid_coords_output':kid_coords_output,\
				'panda_directions_output':panda_directions_output,\
				'panda_idents_output':panda_idents_output,
				'kid_directions_output':kid_directions_output,\
				'kid_idents_output':kid_idents_output,\
				'x_output':x_output,'y_output':y_output,'direction_output':direction_output,\
				'r_output':r_output,'y_outputs':y_outputs,'y_network_ver':y_network_ver})
			savemat(file_name + str(step) + '.mat', dic)
		dic = None
		
		print file_name + str(step) + '.mat'
		print 'step:', step, 'err:',err, 'r:',r_total, 'updates:',network_updates, 'eps:', CHANCE_RAND, 'F1:', np.max(F1), 't:',time.time() - t_start
		if step >= MEM_SZ:
			print 'err_imgnet:', err_imgnet, 'err_imgnet_test:',err_imgnet_test_plot[-1], \
				'class_imgnet_test:',class_err_imgnet_test[-1]
		
		err = 0
		err_imgnet = 0
		r_total = 0
		
		t_start = time.time()
