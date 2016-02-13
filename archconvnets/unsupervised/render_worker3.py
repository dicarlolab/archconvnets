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
from pandac.PandaModules import CullFaceAttrib
import os
from panda3d.core import *
import genthor.model_info as gmi
from archconvnets.unsupervised.rosch_models_collated_reduced import *
from scipy.io import savemat
import sys

t_start = time.time()

IMG_SZ = 32

h = 1

########################################################################### init scene
loadPrcFileData("", "win-size " + str(IMG_SZ) + " " + str(IMG_SZ))
loadPrcFileData("", "window-type offscreen")
loadPrcFileData("", "require-window False")

app = ShowBase()

N_MOVIES = 2500
N_FRAMES = 16
steps = np.linspace(0,180,N_FRAMES)

h = 1

def read_file(func, filepth):
    try:
        out = func(filepth)
    except IOError:
        try:
            out = func(os.path.join(os.getcwd(), filepth))
        except IOError as exc:
            raise exc
    return out

MOVIE_BACKGROUNDS = []; bgtex = []

for i in gmi.BACKGROUNDS:
    MOVIE_BACKGROUNDS.append(i)
    bgtex.append(read_file(app.loader.loadTexture, "/home/darren/.skdata/genthor/resources/backgrounds/" + MOVIE_BACKGROUNDS[-1]))


def get_cannonical(model):
    vertices = np.array(model.getTightBounds())
    
    #Size across full image is 3, so size across 40% is always 1.2 
    initial_scale_factor = max(abs(vertices[0]-vertices[1]))
    canonical_scale = 1.2/initial_scale_factor
    
    #canonical position sets position to 0
    cpos = vertices.mean(0)
    tx = -cpos[1]
    ty = cpos[0]
    tz = cpos[2]
    
    return canonical_scale, tx, ty, tz

zebra = []
zebra_vars = []
s = 1
path = '/home/darren/.skdata/genthor/resources/eggs/'

for f in range(len(objs)):
    t_start = time.time()
    zebra.append(app.loader.loadModel(path + objs[f] + '/' + objs[f]))
    zebra[f].setScale(1,1,1)
    zebra_vars.append(get_cannonical(zebra[f]))
    zebra[f].reparentTo(app.render)
    zebra[f].setScale(s*zebra_vars[f][0],s*zebra_vars[f][0],s*zebra_vars[f][0])
    zebra[f].setPos(-100,-100,-100)


app.camera.setPos(0,0,0)
app.camera.setHpr(0,0,0)

bgnode = app.loader.loadModel('/home/darren/panda_inst/share/panda3d/models/smiley')
bgnode.clearMaterial()
bgnode.clearTexture()
bgnode.setAttrib(CullFaceAttrib.make(CullFaceAttrib.MCullCounterClockwise))
bgnode.reparentTo(app.render)
bgnode.setPos(0, 0, 0)
bgscale=5
bgnode.setScale(bgscale, bgscale, bgscale)

movie_ind = 0

imgs = np.zeros((N_MOVIES, N_FRAMES, 3, IMG_SZ, IMG_SZ),dtype='uint8')
bg_list  = np.zeros(N_MOVIES, dtype='int')
obj_list = np.zeros(N_MOVIES, dtype='int')

bgnode.setTexture(bgtex[bg_list[movie_ind]], 2)

while True:
	bg_list[movie_ind] = np.random.randint(len(bgtex))
	obj_list[movie_ind] = np.random.randint(len(zebra_vars))

	bgnode.setTexture(bgtex[bg_list[movie_ind]], 2)
	bgnode.setHpr(np.random.random()*360,np.random.random()*360,np.random.random()*360)

	obj = obj_list[movie_ind]
	# -1:1, 1.5,4.5 -.8,.8

	y = .6*np.random.random() - .3 + zebra_vars[obj][1]
	z = 1*np.random.random() + 1.5 + zebra_vars[obj][2]
	x = .6*np.random.random() - .3 + zebra_vars[obj][3]

	zebra[obj].setPos(y, z, x)

	h = 360*np.random.random()
	p = 360*np.random.random()
	r = 360*np.random.random()

	hs = np.random.random()*3 - 1 # around x-axis
	ps = 0#np.random.random()*3 - 1
	rs = np.random.random()*3 - 1 # rotating in plane

	success = 0
	
	for frame in range(N_FRAMES):
		zebra[obj].setHpr(h + hs*steps[frame], p + ps*steps[frame], r + rs*steps[frame])

		base.graphicsEngine.render_frame()
		base.screenshot(namePrefix='test2.png',defaultFilename = 0,source=app.win)

		try:
			imgs[movie_ind,frame] = np.ascontiguousarray(np.asarray(PIL.Image.open('test2.png'))[:,:,:3].transpose((2,0,1)))
			success += 1
		except:
			break

	zebra[obj].setPos(-100, -100,-100)
	
	if success == N_FRAMES and (imgs[movie_ind,0] - imgs[movie_ind,1]).max() != 0:
		movie_ind += 1
		print movie_ind
	
	if movie_ind >= N_MOVIES:
		break

print time.time() - t_start

savemat('/home/darren/new_movies3/' + str(sys.argv[1]) + '.mat', {'imgs': imgs, 'bg_list': bg_list, 'obj_list': obj_list})
