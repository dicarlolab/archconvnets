from archconvnets.unsupervised.ntm3.ntm_core import *
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
from multiprocessing.connection import Listener
import os

N_RESET = 10000
IMG_SZ = 32
N_KIDS = 6
N_PANDAS = 6

h = 1

########################################################################### init scene
loadPrcFileData("", "win-size " + str(IMG_SZ) + " " + str(IMG_SZ))
loadPrcFileData("", "window-type offscreen")
loadPrcFileData("", "require-window False")

app = ShowBase()

environ = app.loader.loadModel("/home/darren/panda_inst/share/panda3d/models/environment")
environ.reparentTo(app.render)
environ.setScale(0.25, 0.25, 0.25)
environ.setPos(-8, 42, 0)

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


############################################ render
def render(x,y, direction, kid_coords, panda_coords, kid_directions, panda_directions, filename):
	
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
	

address = ('localhost', PANDA_PORT)
counter = 0

while True:
	listener = Listener(address, authkey='secret password')
	conn = listener.accept()
	while True:
		try:
			x,y, direction, kid_coords, panda_coords, kid_directions, panda_directions, filename = conn.recv()
			render(x,y, direction, kid_coords, panda_coords, kid_directions, panda_directions, filename)
			conn.send('success')
		except:
			break
		
		counter += 1
		if counter >= N_RESET:
			break
			
	conn.close()
	listener.close()
	counter += 1
	if counter >= N_RESET:
		break


print 'restarting'
os.spawnl(os.P_NOWAIT, '/usr/bin/python', 'python', '/home/darren/archconvnets/archconvnets/unsupervised/ntm3/worlds/panda_server.py')

