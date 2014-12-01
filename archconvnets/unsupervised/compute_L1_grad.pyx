cimport numpy as npd
import numpy as np

''' filters: in_channels, filter_sz, filter_sz, n_filters
    imgs: in_channels, img_sz, img_sz, n_imgs
'''
def L1_grad(npd.ndarray[npd.float64_t, ndim=4] F1, npd.ndarray[npd.float64_t, ndim=4] F2, npd.ndarray[npd.float64_t, ndim=4] F3, npd.ndarray[npd.int_t, ndim=4] output_switches3_x, npd.ndarray[npd.int_t, ndim=4] output_switches3_y, npd.ndarray[npd.int_t, ndim=4] output_switches2_x, npd.ndarray[npd.int_t, ndim=4] output_switches2_y, npd.ndarray[npd.int_t, ndim=4] output_switches1_x, npd.ndarray[npd.int_t, ndim=4] output_switches1_y, int s1, int s2, int s3, npd.ndarray[npd.float64_t, ndim=1] pred_cat_sum, npd.ndarray[npd.float64_t, ndim=1] Y_cat_sum, npd.ndarray[npd.float64_t, ndim=4] imgs, int STRIDE1): 
	cdef int N_IMGS = imgs.shape[3]
	cdef int max_output_sz3 = output_switches3_x.shape[1]
	cdef int n3 = output_switches3_x.shape[0]
	cdef int n2 = output_switches2_x.shape[0]
	cdef int n1 = output_switches1_x.shape[0]
	cdef int img
	cdef int channel_ = 0
	cdef int f1_ = 1
	cdef int a1_x_ = 3
	cdef int a1_y_ = 1
	cdef npd.ndarray[npd.float64_t, ndim=4] grad = np.zeros_like(F1)
	cdef int a3_y
	cdef int a3_x
	cdef int f3
	cdef int f2
	cdef int z1
	cdef int z2
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
	for img in range(N_IMGS):
		for a3_x in range(s3):
			for a3_y in range(s3):
				for f3 in range(n3):
					for z1 in range(max_output_sz3):
						for z2 in range(max_output_sz3):
							a3_x_global = output_switches3_x[f3, z1, z2, img] + a3_x
							a3_y_global = output_switches3_y[f3, z1, z2, img] + a3_y
							
							for a2_x in range(s2):
								for a2_y in range(s2):
									for f2 in range(n2):
										# pool2 -> conv2
										a2_x_global = output_switches2_x[f2, a3_x_global, a3_y_global, img] + a2_x
										a2_y_global = output_switches2_y[f2, a3_x_global, a3_y_global, img] + a2_y

										# pool1 -> conv1
										a1_x_global = output_switches1_x[f1_, a2_x_global, a2_y_global, img] * STRIDE1 + a1_x_
										a1_y_global = output_switches1_y[f1_, a2_x_global, a2_y_global, img] * STRIDE1 + a1_y_

										# conv1 -> imgs
										px = imgs[channel_, a1_x_global, a1_y_global, img]
										temp_F_prod_all = F3[f3, f2, a3_x, a3_y] * F2[f2, f1_, a2_x, a2_y] * px

										# supervised term:
										grad[f1_, channel_, a1_x_, a1_y_] -= temp_F_prod_all * Y_cat_sum[img]

										# unsupervised term:
										grad[f1_, channel_, a1_x_, a1_y_] +=  temp_F_prod_all * pred_cat_sum[img];
	return grad

