cimport numpy as npd
import numpy as np

''' filters: in_channels, filter_sz, filter_sz, n_filters
    imgs: in_channels, img_sz, img_sz, n_imgs
'''
def L2_grad(npd.ndarray[npd.float64_t, ndim=4] F1, npd.ndarray[npd.float64_t, ndim=4] F2, npd.ndarray[npd.float64_t, ndim=4] F3, npd.ndarray[npd.float64_t, ndim=4] FL, npd.ndarray[npd.int_t, ndim=4] output_switches3_x, npd.ndarray[npd.int_t, ndim=4] output_switches3_y, npd.ndarray[npd.int_t, ndim=4] output_switches2_x, npd.ndarray[npd.int_t, ndim=4] output_switches2_y, npd.ndarray[npd.int_t, ndim=4] output_switches1_x, npd.ndarray[npd.int_t, ndim=4] output_switches1_y, int s1, int s2, int s3, npd.ndarray[npd.float64_t, ndim=2] pred, npd.ndarray[npd.float64_t, ndim=2] Y, npd.ndarray[npd.float64_t, ndim=4] imgs,  npd.ndarray[npd.float64_t, ndim=5] sigma31, npd.ndarray[npd.int_t, ndim=1] img_cats): 
	cdef int N_C = FL.shape[0]
	cdef int cat
	cdef int N_IMGS = imgs.shape[3]
	cdef int max_output_sz3 = output_switches3_x.shape[1]
	cdef int n3 = output_switches3_x.shape[0]
	cdef int n2 = output_switches2_x.shape[0]
	cdef int n1 = output_switches1_x.shape[0]
	cdef int img
	cdef int f1_
	cdef int a1_x
	cdef int a1_y
	cdef npd.ndarray[npd.float64_t, ndim=4] grad = np.zeros_like(F2)
	cdef int a3_y
	cdef int a3_x
	cdef int f3
	cdef int f2_
	cdef int z1
	cdef int z2
	cdef int channel
	cdef int a2_x_
	cdef int a2_y_
	cdef int a3_x_global
	cdef int a3_y_global
	cdef int a2_x_global
	cdef int a2_y_global
	cdef int a1_x_global
	cdef int a1_y_global	
	cdef float temp_F_prod_all
	cdef float px
	cdef float F31
	cdef float FL31s
		
	for a3_x in range(s3):
		for a3_y in range(s3):
			for a1_x in range(s1):
				for a1_y in range(s1):
					for f3 in range(n3):
						for f1_ in range(n1):
							for channel in range(3):
								for f2_ in range(n2):
									F31 = F3[f3, f2_, a3_x, a3_y] * F1[f1_, channel, a1_x, a1_y]
									for z1 in range(max_output_sz3):
										for z2 in range(max_output_sz3):
											for img in range(N_IMGS):
												FL31s = F31 * FL[img_cats[img], f3, z1, z2] * sigma31[img_cats[img], channel, f1_, a1_x, a1_y]
												for a2_x_ in range(s2):
													for a2_y_ in range(s2):
														# supervised term:
														grad[f2_, f1_, a2_x_, a2_y_] -= FL31s
	
	return grad


