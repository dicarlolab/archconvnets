import os
import time

ind = 0

while True:
	t_start = time.time()
	if (ind % 2) == 0:
		obj_offset = 61
	else:
		obj_offset = 0
	
	os.system('python render_worker3.py ' + str(ind) + ' ' + str(obj_offset))
	print ind, time.time() - t_start
	ind += 1