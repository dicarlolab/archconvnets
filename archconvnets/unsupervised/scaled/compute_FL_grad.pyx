cimport numpy as npd
import numpy as np

''' filters: in_channels, filter_sz, filter_sz, n_filters
    imgs: in_channels, img_sz, img_sz, n_imgs
'''
def FL_grad(npd.ndarray[npd.float64_t, ndim=4] F1, npd.ndarray[npd.float64_t, ndim=4] F2, npd.ndarray[npd.float64_t, ndim=4] F3, npd.ndarray[npd.float64_t, ndim=4] FL, npd.ndarray[npd.int_t, ndim=4] output_switches3_x, npd.ndarray[npd.int_t, ndim=4] output_switches3_y, npd.ndarray[npd.int_t, ndim=4] output_switches2_x, npd.ndarray[npd.int_t, ndim=4] output_switches2_y, npd.ndarray[npd.int_t, ndim=4] output_switches1_x, npd.ndarray[npd.int_t, ndim=4] output_switches1_y, int s1, int s2, int s3, npd.ndarray[npd.float64_t, ndim=2] pred, npd.ndarray[npd.float64_t, ndim=2] Y, npd.ndarray[npd.float64_t, ndim=4] imgs,  npd.ndarray[npd.float64_t, ndim=5] sigma31, npd.ndarray[npd.int_t, ndim=1] img_cats): 
	cdef int N_C = FL.shape[0]
	cdef int cat_
	cdef int N_IMGS = imgs.shape[3]
	cdef int max_output_sz3 = output_switches3_x.shape[1]
	cdef int n3 = output_switches3_x.shape[0]
	cdef int n2 = output_switches2_x.shape[0]
	cdef int n1 = output_switches1_x.shape[0]
	cdef int img
	cdef int channel
	cdef int f1
	cdef int a1_x
	cdef int a1_y
	cdef npd.ndarray[npd.float64_t, ndim=4] grad = np.zeros_like(FL)
	cdef int a3_y
	cdef int a3_x
	cdef int f3_
	cdef int f2
	cdef int z1_
	cdef int z2_
	cdef int a2_x
	cdef int a2_y
	cdef int a3_x_global
	cdef int a3_y_global
	cdef int a2_x_global
	cdef int a2_y_global
	cdef int a1_x_global
	cdef int a1_y_global	
	cdef float temp_F_prod_all
	cdef float px
	cdef float F32
	cdef float F321
	
	for a3_x in range(s3):
		for a3_y in range(s3):
			for a2_x in range(s2):
				for a2_y in range(s2):
					for f3_ in range(n3):
						for f2 in range(n2):
							for f1 in range(n1):
								F32 = F3[f3_, f2, a3_x, a3_y] * F2[f2, f1, a2_x, a2_y]
								for a1_x in range(s1):
									for a1_y in range(s1):
										for channel in range(3):
											for img in range(N_IMGS):
												F321s = F32 * F1[f1, channel, a1_x, a1_y] * sigma31[img_cats[img], channel, f1, a1_x, a1_y]
												for z1_ in range(max_output_sz3):
													for z2_ in range(max_output_sz3):
														# supervised term:
														grad[img_cats[img], f3_, z1_, z2_] -= F321s
	return grad


