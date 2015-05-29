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
import pickle as pk

IMG_SZ = 92

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


#################################### 
N_BATCHES = 9374
BATCH_SZ = 256

labels = np.zeros(BATCH_SZ, dtype='int')
imgs_new = np.zeros((3*92*92, BATCH_SZ), dtype='uint8')

write_batch = 1
img_i_global = 0
filenames = [None]*BATCH_SZ
t_start = time.time()
data_mean = np.zeros((3*92*92))

z = np.load('/export/batch_storage2/batch128_img138_full/batches.meta')
z['num_cases_per_batch'] = BATCH_SZ
z['num_vis'] = 3*92*92

for batch in range(1,N_BATCHES+1):
	for img_i in range(128):
		img_local = img_i_global % BATCH_SZ
		
		##### render
		r = 0
		while r == 0:
			x,y, direction, kid_coords, panda_coords, kid_directions, panda_directions = init_pos_vars()
			xo = copy.deepcopy(x)
			yo = copy.deepcopy(y)
			kid_coordso = copy.deepcopy(kid_coords)
			panda_coordso = copy.deepcopy(panda_coords)
			kid_directionso = copy.deepcopy(kid_directions)
			panda_directionso = copy.deepcopy(panda_directions)
			
			r, x2,y2, direction2, kid_coords2, panda_coords2, kid_directions2, panda_directions2 = move(x,y, direction, kid_coords, panda_coords, kid_directions, panda_directions)
			
		img = render(xo,yo, direction, panda, kid, kid_coordso, panda_coordso, kid_directionso, panda_directionso)
		
		imgs_new[:,img_local] = img[0].ravel()
		data_mean += imgs_new[:,img_local]
		labels[img_local] = (r + 1)/2
		filenames[img_local] = str(img_i_global)
		if img_local == (BATCH_SZ-1):
			print 'writing', write_batch, time.time() - t_start
			t_start = time.time()
			file = open('/export/storage/panda_environ_cudaconvnet/data_batch_' + str(write_batch-1),'w')
			pk.dump({'data':imgs_new, 'mean': imgs_new.mean(1)[:,np.newaxis], 'labels': labels, 'batch_label': str(write_batch)}, file)
			file.close()
			write_batch += 1
			
			#### meta batch
			z['data_mean'] = copy.deepcopy(data_mean / img_i_global)
			file = open('/export/storage/panda_environ_cudaconvnet/batches.meta','w')
			pk.dump(z, file)
			file.close()
		
		img_i_global += 1
