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
from pandac.PandaModules import CullFaceAttrib
import os
from panda3d.core import *
import genthor.model_info as gmi
from archconvnets.unsupervised.rosch_models_collated import *

IMG_SZ = 32

h = 1

########################################################################### init scene
loadPrcFileData("", "win-size " + str(IMG_SZ) + " " + str(IMG_SZ))
loadPrcFileData("", "window-type offscreen")
loadPrcFileData("", "require-window False")
print 'starting'
app = ShowBase()
print 'done'
PANDA_SZ = 2.5
KID_SZ = 1.25
ROOM_SZ = 9
ROOM_SZ_MV = 10
N_KIDS = 6
N_PANDAS = 6

ROT_RATE = 12
MOV_RATE = .75
N_FRAMES = 16
steps = np.linspace(0,180,N_FRAMES)

h = 1

def read_file(func, filepth):
    """ Returns func(filepath), first trying absolute path, then
    relative."""

    try:
        out = func(filepth)
    except IOError:
        try:
            out = func(os.path.join(os.getcwd(), filepth))
        except IOError as exc:
            raise exc
    return out

MOVIE_BACKGROUNDS = []; bgtex = []
print 'loading backgrounds'
for i in gmi.BACKGROUNDS:
    MOVIE_BACKGROUNDS.append(i)
    bgtex.append(read_file(app.loader.loadTexture, "/home/darren/.skdata/genthor/resources/backgrounds/" + MOVIE_BACKGROUNDS[-1]))
print 'done'

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
path = '.skdata/genthor/resources/eggs/'
for f in range(len(objs)):
    t_start = time.time()
    zebra.append(app.loader.loadModel(path + objs[f] + '/' + objs[f]))
    zebra[f].setScale(1,1,1)
    zebra_vars.append(get_cannonical(zebra[f]))
    zebra[f].reparentTo(app.render)
    zebra[f].setScale(s*zebra_vars[f][0],s*zebra_vars[f][0],s*zebra_vars[f][0])
    zebra[f].setPos(-100,-100,-100)
    #print f, time.time() - t_start

app.camera.setPos(0,0,0)
app.camera.setHpr(0,0,0)

bgnode = app.loader.loadModel('panda_inst/share/panda3d/models/smiley')
bgnode.clearMaterial()
bgnode.clearTexture()
bgnode.setAttrib(CullFaceAttrib.make(CullFaceAttrib.MCullCounterClockwise))
bgnode.reparentTo(app.render)
bgnode.setPos(0, 0, 0)
bgscale=5
bgnode.setScale(bgscale, bgscale, bgscale)

N_MOVIES = 1

t_start = time.time()

movie_ind = 0

imgs = np.zeros((N_MOVIES, N_FRAMES, 3, IMG_SZ, IMG_SZ),dtype='single')

while True:
    bgnode.setTexture(bgtex[np.random.randint(len(bgtex))], 2)
    bgnode.setHpr(np.random.random()*360,np.random.random()*360,np.random.random()*360)
    
    obj = np.random.randint(len(zebra_vars))
    # -1:1, 1.5,4.5 -.8,.8
    
    y = 1.6*np.random.random() - .8 + zebra_vars[obj][1]
    z = 3*np.random.random() + 1.5 + zebra_vars[obj][2]
    x = 1.6*np.random.random() - .8 + zebra_vars[obj][3]
    
    zebra[obj].setPos(y, z, x)
    
    h = 360*np.random.random()
    p = 360*np.random.random()
    r = 360*np.random.random()
    
    hs = np.random.random()*3 - 1
    ps = np.random.random()*3 - 1
    rs = np.random.random()*3 - 1
    
    sz = 9*2*2
    fig = figure(1,figsize=(sz,sz))
    
    for frame in range(N_FRAMES):
        zebra[obj].setHpr(h + hs*steps[frame], p + ps*steps[frame], r + rs*steps[frame])
        
        base.graphicsEngine.render_frame()
        base.screenshot(namePrefix='test.png',defaultFilename = 0,source=app.win)
        
        try:
            print np.ascontiguousarray(np.asarray(PIL.Image.open('test.png'))[:,:,:3].transpose((2,0,1))).shape
            imgs[movie,frame] = np.ascontiguousarray(np.asarray(PIL.Image.open('test.png'))[:,:,:3].transpose((2,0,1)))
            #fig.add_subplot(N_FRAMES,1,frame+1)
            #imshow(z[0].transpose((1,2,0)),interpolation='nearest');axis('off');
            movie_ind += 1
        except:
            print obj
            break
    
    zebra[obj].setPos(-100, -100,-100)
    print '.........', movie_ind
    if movie_ind >= N_MOVIES:
        break

print time.time() - t_start
