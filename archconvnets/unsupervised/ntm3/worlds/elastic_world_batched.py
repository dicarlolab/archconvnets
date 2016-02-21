import copy
import numpy as np
from ntm_core import *

def check_collision(x_possible, x_sz):
    # check collision with other object
    top_edge = x_possible[:,0] - x_sz/2
    bottom_edge = x_possible[:,0] + x_sz/2
    left_edge = x_possible[:,1] - x_sz/2
    right_edge = x_possible[:,1]+ x_sz/2
    
    return ((bottom_edge[0] > top_edge[1]) and (bottom_edge[0] < bottom_edge[1]) and (right_edge[0] > left_edge[1]) and (right_edge[0] < right_edge[1])) or \
       ((bottom_edge[0] > top_edge[1]) and (bottom_edge[0] < bottom_edge[1]) and (left_edge[0] > left_edge[1]) and (left_edge[0] < right_edge[1])) or \
       ((top_edge[0] > top_edge[1]) and (top_edge[0] < bottom_edge[1]) and (left_edge[0] > left_edge[1]) and (left_edge[0] < right_edge[1])) or \
       ((top_edge[0] > top_edge[1]) and (top_edge[0] < bottom_edge[1]) and (right_edge[0] > left_edge[1]) and (right_edge[0] < right_edge[1]))

def update_state(x, v, m, x_sz):
    n_objs = x.shape[0]
    
    x_possible = x + v
    
    if check_collision(x_possible, x_sz):
            v_prev = copy.deepcopy(v)
            #print 'collision'

            v[0] = (v_prev[0]*(m[0] - m[1]) + 2*m[1]*v_prev[1])/(m[0] + m[1])
            v[1] = (v_prev[1]*(m[1] - m[0]) + 2*m[0]*v_prev[0])/(m[0] + m[1])
    
    x += v
    
    # check collision with edge of room
    for obj in range(n_objs):
        for dim in range(2):
            if x[obj][dim] > 1 or x[obj][dim] < 0:
                v[obj][dim] = -v[obj][dim]
        x[obj][x[obj] > 1] = 1
        x[obj][x[obj] < 0] = 0
    
    return x,v

def init_state():
    n_objs = 2
    v_scale = 2e-1
    x_scale = .2
    
    x_sz = x_scale * np.ones(n_objs)
    v = v_scale * (np.random.random((BATCH_SZ, n_objs, 2)) - .5)
    m = np.random.random((BATCH_SZ, n_objs))
    
    # sample starting positions, but do not allow objects to be overlapping
    x = np.random.random((BATCH_SZ, n_objs, 2))
    for batch in range(BATCH_SZ):
		while check_collision(x[batch], x_sz):
			x[batch] = np.random.random((n_objs, 2))
        
    return x, v, m, x_sz, 32

def crop_ind(x, im_sz):
    if x >= im_sz:
        x = im_sz-1
    if x < 0:
        x = 0
    return x

def show_state(x, v, m, x_sz, im_sz):
    n_objs = x.shape[0]
    im = np.zeros((im_sz,im_sz))
    
    for obj in range(n_objs):
        x1 = crop_ind(np.round((x[obj][0] - x_sz[obj]/2)*im_sz), im_sz)
        x2 = crop_ind(np.round((x[obj][0] + x_sz[obj]/2)*im_sz), im_sz)
        
        y1 = crop_ind(np.round((x[obj][1] - x_sz[obj]/2)*im_sz), im_sz)
        y2 = crop_ind(np.round((x[obj][1] + x_sz[obj]/2)*im_sz), im_sz)

        im[x1:x2][:,y1:y2] = m[obj] + .5
    return im

def generate_imgs(n_imgs=34):
    inputs = np.zeros((n_imgs, BATCH_SZ, 3, 32,32), dtype='single')
    
    x, v, m, x_sz, im_sz = init_state()

    for frame in range(n_imgs):
		for batch in range(BATCH_SZ):
			inputs[frame, batch] = show_state(x[batch], v[batch], m[batch], x_sz, im_sz) - .5
			x[batch], v[batch] = update_state(x[batch], v[batch], m[batch], x_sz)
    
    targets = copy.deepcopy(inputs)
    targets = targets.reshape((n_imgs, BATCH_SZ, 3*32*32,1))
    
    inputs = inputs.reshape((n_imgs, BATCH_SZ, 3, 32, 32))
    
    return inputs, targets
	
def generate_latents(EPOCH_LEN=32, T_AHEAD=2):
	n_imgs = EPOCH_LEN + T_AHEAD

	x, v, m, x_sz, im_sz = init_state()

	inputs = np.zeros(np.concatenate(((n_imgs,), x.shape)), dtype='single')

	for frame in range(n_imgs):
		inputs[frame] = copy.deepcopy(x)
		x, v = update_state(x, v, m, x_sz)

	inputs = inputs.reshape((n_imgs, np.prod(x.shape), 1))

	return inputs, inputs