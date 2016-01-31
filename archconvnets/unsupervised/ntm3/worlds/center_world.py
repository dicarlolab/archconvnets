import copy
import numpy as np


def update_state(x, v, m, x_sz):
    n_objs = x.shape[0]
    
    x += v
    
    return x,v

def init_state(EPOCH_LEN):
	n_objs = 1
	v_scale = 2e-1
	x_scale = .2
	im_sz = 32

	x_sz = x_scale * np.ones(n_objs)
	m = np.random.random(n_objs)

	# sample starting positions, but do not allow objects to be overlapping
	x = np.random.random((n_objs, 2))

	v = (1 - 2*x) / EPOCH_LEN

	return x, v, m, x_sz, im_sz

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

def generate_imgs(EPOCH_LEN=32):
    n_imgs = EPOCH_LEN
    
    inputs = np.zeros((n_imgs, 3, 32,32), dtype='single')
    
    x, v, m, x_sz, im_sz = init_state(EPOCH_LEN)

    for frame in range(n_imgs):
        inputs[frame] = show_state(x, v, m, x_sz, im_sz) - .5
        x, v = update_state(x, v, m, x_sz)
    
    targets = copy.deepcopy(inputs)
    targets = targets.reshape((n_imgs, 3*32*32,1))
    
    inputs = inputs.reshape((n_imgs, 1, 3, 32, 32))
    
    return inputs, targets
	
def generate_latents(EPOCH_LEN=32):
	n_imgs = EPOCH_LEN

	x, v, m, x_sz, im_sz = init_state(EPOCH_LEN)

	inputs = np.zeros(np.concatenate(((n_imgs,), x.shape)), dtype='single')

	for frame in range(n_imgs):
		inputs[frame] = copy.deepcopy(x)
		x, v = update_state(x, v, m, x_sz)

	inputs = inputs.reshape((n_imgs, np.prod(x.shape), 1))

	return inputs, inputs