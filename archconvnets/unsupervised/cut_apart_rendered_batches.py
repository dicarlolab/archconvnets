import time
import numpy as np
import Image
from scipy.io import savemat, loadmat

targets = np.zeros((2500, 16-3, 3, 16, 16), dtype='uint8')

for file in range(443,540):
	print file
	z = loadmat('/home/darren/new_movies3/' + str(file) + '.mat')
	inputs = z['imgs'][:,:3]
	for img in range(2500):
		for frame in range(3,16):
			targets[img,frame-3] = np.asarray(Image.fromarray(z['imgs'][img,frame].transpose((1,2,0))).resize((16,16))).transpose((2,0,1))
	savemat('/home/darren/new_movies3_cut/' + str(file) + '.mat', {'bg_list':z['bg_list'], 'obj_list':z['obj_list'], 'inputs':inputs,'targets':targets})