import os
import time

ind = 13

obj_offset = 61

while True:
	t_start = time.time()
	if obj_offset == 0:
		obj_offset = 61
	else:
		obj_offset = 0
	
	os.system('python render_worker.py ' + str(ind) + ' ' + str(obj_offset))
	print ind, time.time() - t_start
	ind += 1