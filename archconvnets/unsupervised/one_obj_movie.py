import genthor; import genthor.datasets as gd;
from matplotlib.pylab import *
import numpy as np
import datasets as gd
import random
import math
import scipy
from PIL import Image, ImageSequence
from images2gif import writeGif
import sys, os
import genthor.model_info as model_info
import copy
import itertools
import time
import numpy as np
import genthor.datasets as gd
import genthor.model_info as gmi
from scipy.io import savemat

import tabular as tb 
import os

import pyll
choice = pyll.scope.choice
uniform = pyll.scope.uniform
loguniform = pyll.scope.loguniform
import pyll.stochastic as stochastic

from yamutils.basic import dict_inverse

try:
    from collections import OrderedDict
except ImportError:
    print "Python 2.7+ OrderedDict collection not available"
    try:
        from ordereddict import OrderedDict
    except ImportError:
        raise ImportError("OrderedDict not available")

t_start = time.time()

MOVIE_OBJS = OrderedDict([('animal', ['weimaraner','lo_poly_animal_ELE_AS1','lo_poly_animal_TRTL_B','lo_poly_animal_TRANTULA']),
  ('boats', ['MB29346', 'MB29766', 'MB29762', 'MB27061']),
  ('cars', ['MB28490', 'MB31620', 'MB31417', 'MB29183']),
  ('chairs', ['MB29822', 'MB28186', 'MB27704', 'MB27139']),
  ('faces', ['face1', 'face2', 'face3', 'face4']),
  ('fruits', ['single_apple', 'single_pumpkin', 'single_banana', 'single_tomato']),
  ('planes', ['MB29798', 'MB29050','MB28259', 'MB27215']),
  ('tables', ['antique_furniture_item_18', 'office_equipment_73_4', 'antique_furniture_item_48', 'MB30922'])
])
cat_list = ['animal', 'boats','cars','chairs','faces','fruits','planes','tables']
#MOVIE_BACKGROUNDS = [gmi.BACKGROUNDS[0], gmi.BACKGROUNDS[10], gmi.BACKGROUNDS[50], gmi.BACKGROUNDS[100], gmi.BACKGROUNDS[120]]
MOVIE_BACKGROUNDS = []
for i in gmi.BACKGROUNDS:
	MOVIE_BACKGROUNDS.append(i)


get_image_id = gd.get_image_id
def get_two_movie_obj_latents(tdict, models, categories):
    rng = np.random.RandomState(seed=tdict['seed'])
    latents = []
    tname = tdict['name']
    template = tdict['template']
    n_frames = tdict['n_frames_obj']
    n_frames_stagger = tdict['n_frames_stagger']
    if tdict['reverse_front_back'] == 1:
        front_model = models[1]
        back_model = models[0]
        print "switch"
    else:
        front_model = models[0]
        back_model = models[1]
    l = stochastic.sample(template, rng)
    print l['bgname']
    l['obj'] = [front_model, back_model]
    l['category'] = [categories[front_model][0],
                     categories[back_model][0]]
    bgphi = 180*np.cos(np.linspace(0,l['bgphi_freq']*math.pi/12,n_frames+n_frames_stagger) + 3*180*l['bgphi_phase'])
    bgpsi = 180*np.cos(np.linspace(0,l['bgpsi_freq']*math.pi/12,n_frames+n_frames_stagger) + 3*180*l['bgpsi_phase'])
    ############################3
    # determine positions for each frame in movie
    
    # front obj:
    # start position for object (will always start at the edge of an image)
    front_rxy = np.linspace(0,91,n_frames) + 3*180*l['front_rxy_phase']
    front_ryz = np.linspace(0,91,n_frames) + 3*180*l['front_ryz_phase']
    front_rxz = np.linspace(0,91,n_frames) + 3*180*l['front_rxz_phase']
    phi_cos = 2*math.pi*l['front_phase']
    phi_cos_z = 2*math.pi*l['front_phase_z']
    front_freq = l['front_freq']
    front_z_freq = l['front_z_freq']
    alpha = 2.5
    if l['front_xy_edge'] == 1: # y is -1 or 1
        y_start = alpha*l['front_edge_side']
        y_end = -y_start;
        x_start = l['front_edge_start']*2 - 1
        x_end = l['front_edge_end']*2 - 1
        front_x_steps = np.linspace(x_start, x_end, n_frames)
        front_y_steps = np.linspace(y_start, y_end, n_frames)
        front_x_steps += .3*np.cos(2*front_freq*math.pi*np.arange(n_frames)/n_frames + phi_cos)
    else:
        x_start = alpha*l['front_edge_side']
        x_end = -x_start;
        y_start = l['front_edge_start']*2 - 1
        y_end = l['front_edge_end']*2 - 1
        front_x_steps = np.linspace(x_start, x_end, n_frames)
        front_y_steps = np.linspace(y_start, y_end, n_frames)
        front_y_steps += .3*np.cos(2*front_freq*math.pi*np.arange(n_frames)/n_frames + phi_cos)
    front_z_steps = l['front_z_amp']*np.cos(2*front_z_freq*math.pi*np.arange(n_frames)/n_frames + phi_cos_z)
    
    #############
    # follow the front object? or have a random, indepdendent trajectory?
    if l['follow'] == True:
        back_rxy = front_rxy + l['rxy_offset'] 
        back_ryz = front_ryz + l['ryz_offset'] 
        back_rxz = front_rxz + l['rxz_offset']
        phi_cos += l['phase_offset']
        phi_cos_z += l['phase_z_offset']
        front_freq += l['freq_offset']
        front_z_freq += l['z_freq_offset']
        if l['front_xy_edge'] == 1: # y is -1 or 1
            x_start += l['edge_start_offset']
            x_end += l['edge_start_offset']
            back_x_steps = np.linspace(x_start, x_end, n_frames)
            back_y_steps = np.linspace(y_start, y_end, n_frames)
            back_x_steps += .3*np.cos(2*front_freq*math.pi*np.arange(n_frames)/n_frames + phi_cos)
        else:
            y_start += l['edge_start_offset']
            y_end += l['edge_start_offset']
            back_x_steps = np.linspace(x_start, x_end, n_frames)
            back_y_steps = np.linspace(y_start, y_end, n_frames)
            back_y_steps += .3*np.cos(2*front_freq*math.pi*np.arange(n_frames)/n_frames + phi_cos)
        back_z_steps = l['front_z_amp']*np.cos(2*front_z_freq*math.pi*np.arange(n_frames)/n_frames + phi_cos_z)
        
        ##########
        # create a random, independent trajectory
    else:
        # back obj
        back_rxy = np.linspace(0,91,n_frames) + 3*180*l['back_rxy_phase']
        back_ryz = np.linspace(0,91,n_frames) + 3*180*l['back_ryz_phase']
        back_rxz = np.linspace(0,91,n_frames) + 3*180*l['back_rxz_phase']
        phi_cos = 2*math.pi*l['back_phase']
        phi_cos_z = 2*math.pi*l['back_phase_z']
        back_freq = l['back_freq']
        back_z_freq = l['back_z_freq']
        if l['back_xy_edge'] == 1: # y is -1 or 1
            y_start = alpha*l['back_edge_side']
            y_end = -y_start;
            x_start = l['back_edge_start']*2 - 1
            x_end = l['back_edge_end']*2 - 1
            back_x_steps = np.linspace(x_start, x_end, n_frames)
            back_y_steps = np.linspace(y_start, y_end, n_frames)
            back_x_steps += .3*np.cos(2*back_freq*math.pi*np.arange(n_frames)/n_frames + phi_cos)
        else:
            x_start = alpha*l['back_edge_side']
            x_end = -x_start;
            y_start = l['back_edge_start']*2 - 1
            y_end = l['back_edge_end']*2 - 1
            back_x_steps = np.linspace(x_start, x_end, n_frames)
            back_y_steps = np.linspace(y_start, y_end, n_frames)
            back_y_steps += .3*np.cos(2*back_freq*math.pi*np.arange(n_frames)/n_frames + phi_cos)
        back_z_steps = l['back_z_amp']*np.cos(2*back_z_freq*math.pi*np.arange(n_frames)/n_frames + phi_cos_z)
    
    ####################
    # generate the templates for each frame
    for frame in range(n_frames + n_frames_stagger):
        l['bgphi'] = bgphi[frame]
        l['bgpsi'] = bgpsi[frame]
        if (frame < n_frames) and l['show_front_object'] == True:
            l['front_rxz'] = front_rxz[frame]
            l['front_rxy'] = front_rxy[frame]
            l['front_ryz'] = front_ryz[frame]
            l['front_ty'] = front_y_steps[frame]
            l['front_tz'] = front_x_steps[frame]
            l['front_tx'] = front_z_steps[frame]
        else:
            l['front_rxz'] = front_rxz[0]
            l['front_rxy'] = front_rxy[0]
            l['front_ryz'] = front_ryz[0]
            l['front_ty'] = front_y_steps[0]
            l['front_tz'] = front_x_steps[0]
            l['front_tx'] = front_z_steps[0]
        if (frame >= n_frames_stagger) and l['show_back_object'] == True:
            l['back_rxz'] = back_rxz[frame - n_frames_stagger]
            l['back_rxy'] = back_rxy[frame - n_frames_stagger]
            l['back_ryz'] = back_ryz[frame - n_frames_stagger]
            l['back_ty'] = back_y_steps[frame - n_frames_stagger]
            l['back_tz'] = back_x_steps[frame - n_frames_stagger]
            l['back_tx'] = back_z_steps[frame - n_frames_stagger]
        else:
            l['back_rxz'] = back_rxz[0]
            l['back_rxy'] = back_rxy[0]
            l['back_ryz'] = back_ryz[0]
            l['back_ty'] = back_y_steps[0]
            l['back_tz'] = back_x_steps[0]
            l['back_tx'] = back_z_steps[0]
        
        l['id'] = get_image_id(l)                                   
        rd = float(l['back_rd'])
        ad = float(l['back_ad'])
        yd = rd * np.cos(ad)
        zd = rd * np.sin(ad)
        rec = (l['bgname'],
               float(l['bgphi']),
               float(l['bgpsi']),
               float(l['bgscale']),
               l['category'],
               l['obj'],
               [float(l['front_ryz']), float(l['back_ryz'])],
               [float(l['front_rxz']), float(l['back_rxz'])],
               [float(l['front_rxy']), float(l['back_rxy'])],
               [float(l['front_ty']), float(l['back_ty'])],
               [float(l['front_tz']), float(l['back_tz'])],
               [float(l['front_tx']-2), float(l['back_tx']+2)],
               [float(l['front_s']), float(l['back_s'])],
               [None, None],
               [None, None],
               tname,
               l['id'])
        latents.append(rec)
            
    return latents

IM_SZ = 32 #64
preproc = {'dtype':'float32', 'size':(IM_SZ, IM_SZ, 3), 'normalize':False, 'mode':'RGB'}
class MovieDataset(gd.GenerativeBase):
    check_penetration = False
    def _get_meta(self):
        models = self.models
        templates = self.templates
        use_canonical = self.use_canonical
        internal_canonical = True #self.internal_canonical
        
        latents = []
        rng = np.random.RandomState(seed=1)
        model_categories = self.model_categories
    
        for tdict in templates:
            nobjects = tdict['n_objects']
            if nobjects == 0:
                latents.extend(get_noobj_latents(tdict))
            elif nobjects == 1:
                latents.extend(get_oneobj_latents(tdict, models, model_categories))
            elif nobjects == 2:
                latents.extend(get_two_movie_obj_latents(tdict, models, model_categories))
        
        ids = [_x[-1] for _x in latents]
        #assert len(ids) == len(set(ids))
        idlen = max(map(len, ids))
        tnames = [_x[-2] for _x in latents]
        tnamelen = max(map(len, tnames))

        meta = tb.tabarray(records=latents, names = ['bgname','bgphi','bgpsi','bgscale',
                                                     'category','obj','ryz','rxz','rxy','ty',
                                                     'tz','tx','s','texture','texture_mode','tname','id'], 
                           formats=['|S20', 'float', 'float', 'float'] + \
                                  ['|O8']*11 +  ['|S%s' % tnamelen, '|S%s' % idlen])
            
        if internal_canonical:
            meta = meta.addcols([np.ones((len(meta),))], names = ['internal_canonical'])
        else:
            meta = meta.addcols([np.zeros((len(meta),))], names = ['internal_canonical'])
        
        n_objs = map(len, meta['obj'])
        meta = meta.addcols([n_objs], names = ['n_objects'])
        return meta 

n_frames = 25
n_frames_stagger = 1

templates_g = [
                 {'n_objects': 2,
                  'n_frames_obj': n_frames,
				  'n_frames_stagger': n_frames_stagger,
                  'name': 'TwoObject', 
                  'reverse_front_back': choice([1, 0]),
                  'template': {'bgname': choice(MOVIE_BACKGROUNDS),
                    'front_phase': uniform(0,1),
                    'front_xy_edge': choice([0,1]),
                    'front_edge_side': choice([-1,1]),
                    'front_edge_start': uniform(0,1),
                    'front_edge_end': uniform(0,1),
                    'front_bgphi_phase': uniform(0,1),
                    'front_bgpsi_phase': uniform(0,1),
                    'front_rxy_phase': uniform(0,1),
                    'front_rxz_phase': uniform(0,1),
                    'front_ryz_phase': uniform(0,1),
                    'front_freq': 1,
                    'front_z_amp': 2,
                    'back_z_amp': 2,
                    'front_z_freq': 1,
                    'front_phase_z': uniform(0,1),
                    'back_phase_z': uniform(0,1),
                    'back_phase': uniform(0,1),
                    'back_xy_edge': choice([0,1]),
                    'back_edge_side': choice([-1,1]),
                    'back_edge_start': uniform(0,1),
                    'back_edge_end': uniform(0,1),
                    'back_bgphi_phase': uniform(0,1),
                    'back_bgpsi_phase': uniform(0,1),
                    'back_rxy_phase': uniform(0,1),
                    'back_rxz_phase': uniform(0,1),
                    'back_ryz_phase': uniform(0,1),
                    'back_freq': 1,
                    'back_z_freq': 1,
                    'bgphi_freq': 1,
                    'bgpsi_freq': 1,
                    'bgphi_phase': uniform(0,1),
                    'bgpsi_phase': uniform(0,1),
                    'front_x_offset': 0,
                    'back_x_offset': 0,
                    'front_s': uniform(1, 2),
                     'back_s': uniform(1, 2),  
                     'bgscale': 1.,
                     'back_ad': uniform(0, 2 * np.pi),
                     'back_rd': uniform(.3, .85),
                     'front_tx': 1.5,
                     'back_txd': 2.5,
                    
                    'show_front_object': True, # ex. for 0 object movies
                    'show_back_object': False,
                    
                     'follow': False, # should object 2 follow object 1?
                     'rxy_offset': .1,
                     'ryz_offset': .1,
                     'rxz_offset': .1,
                     'phase_offset': .1,
                     'phase_z_offset': .1,
                     'freq_offset': .1,
                     'z_freq_offset': .1,
                     'edge_start_offset': .1,
                     }}
               ]
               

scales = np.ones((8,4))*2
seed_offset = 666

# rsync -avz rotating_objs 18.93.13.151:/home/darren
if __name__ == '__main__':
	n_t = 10 + n_frames_stagger
	pad = (n_frames - n_t)/2

	#imgs = np.zeros((11*N_MOVIES, IM_SZ, IM_SZ, 3),dtype='single')
	
	#for i in range(N_MOVIES):
	i = 2319
	while True:
		t_start = time.time()
		os.system('rm .skdata/genthor/cache/one_obj_32movie_1a* -r')
		
		cat_i = np.random.randint(8)
		cat_j = np.random.randint(8)
		
		obj_i = np.random.randint(4)
		obj_j = np.random.randint(4)
		
		class one_obj_32movie_1a(MovieDataset):
			models = [MOVIE_OBJS[cat_list[cat_i]][obj_i], MOVIE_OBJS[cat_list[cat_j]][obj_j]]
			model_categories = dict_inverse(MOVIE_OBJS)
			model_categories = {models[0]: model_categories[models[0]], models[1]: model_categories[models[1]]}
			templates = copy.deepcopy(templates_g)
			templates[0]['template']['front_s'] = scales[cat_i,obj_i]
			templates[0]['template']['back_s'] = scales[cat_j,obj_j]
			templates[0]['seed'] = 1034423424 + np.random.randint(10344234244)

		
		dataset = one_obj_32movie_1a()
		try:
			imgs = copy.deepcopy(dataset.get_images(preproc)[pad:pad+n_t][::-1]).transpose((0,3,1,2))[:,np.newaxis]
			savemat('rotating_objs32_25t/imgs' + str(i) + '.mat',{'imgs':imgs})
			i+=1
			print "%i elapsed time %f" % (i, time.time()-t_start)
		except:
			print 'failed'
