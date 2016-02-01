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
from multiprocessing.connection import Client

IMG_SZ = 32
PANDA_SZ = 2.5
KID_SZ = 1.25
ROOM_SZ = 9
ROOM_SZ_MV = 10
N_KIDS = 6
N_PANDAS = 6

ROT_RATE = 12
MOV_RATE = .75

h = 1


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
def render(x,y, direction, kid_coords, panda_coords, kid_directions, panda_directions, filename='tmp2'):
	address = ('localhost', 6000)
	success = False
	while True:
		try:
			conn = Client(address, authkey='secret password')
			conn.send([x,y, direction, kid_coords, panda_coords, kid_directions, panda_directions, filename])
			msg = conn.recv()
			conn.close()
			success = True
		except:
			print 'failed to connect'
			time.sleep(2)
		if success:
			break
	return msg

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

