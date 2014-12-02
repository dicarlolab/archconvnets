cimport numpy as npd
import numpy as np

''' filters: in_channels, filter_sz, filter_sz, n_filters
    imgs: in_channels, img_sz, img_sz, n_imgs
'''
def L1_grad(npd.ndarray[npd.float64_t, ndim=4] F1, npd.ndarray[npd.float64_t, ndim=4] F2, npd.ndarray[npd.float64_t, ndim=4] F3, npd.ndarray[npd.float64_t, ndim=4] FL, npd.ndarray[npd.int_t, ndim=4] output_switches3_x, npd.ndarray[npd.int_t, ndim=4] output_switches3_y, npd.ndarray[npd.int_t, ndim=4] output_switches2_x, npd.ndarray[npd.int_t, ndim=4] output_switches2_y, npd.ndarray[npd.int_t, ndim=4] output_switches1_x, npd.ndarray[npd.int_t, ndim=4] output_switches1_y, int s1, int s2, int s3, npd.ndarray[npd.float64_t, ndim=2] pred, npd.ndarray[npd.float64_t, ndim=2] Y, npd.ndarray[npd.float64_t, ndim=4] imgs): 
	cdef int N_C = FL.shape[0]
	cdef int cat
	cdef int N_IMGS = imgs.shape[3]
	cdef int max_output_sz3 = output_switches3_x.shape[1]
	cdef int n3 = output_switches3_x.shape[0]
	cdef int n2 = output_switches2_x.shape[0]
	cdef int n1 = output_switches1_x.shape[0]
	cdef int img
	cdef int channel_
	cdef int f1_
	cdef int a1_x_
	cdef int a1_y_
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
	cdef float F32
	cdef float FL32
	cdef npd.ndarray[npd.int_t, ndim=4] output_switches3_xt
	cdef npd.ndarray[npd.int_t, ndim=4] output_switches3_yt
	#cdef float grad_s
	#cdef float grad_uns
		
	for a3_x in range(s3):
		output_switches3_xt = output_switches3_x + a3_x
		for a3_y in range(s3):
			output_switches3_yt = output_switches3_y + a3_y
			for f1_ in range(n1):
				for f3 in range(n3):
					for f2 in range(n2):
						for a2_x in range(s2):
							for a2_y in range(s2):
								F32 = F3[f3, f2, a3_x, a3_y] * F2[f2, f1_, a2_x, a2_y]
								for cat in range(N_C):
									for z1 in range(max_output_sz3):
										for z2 in range(max_output_sz3):
											FL32 = F32 * FL[cat, f3, z1, z2]
											for img in range(N_IMGS):
												# pool3 -> conv3
												a3_x_global = output_switches3_xt[f3, z1, z2, img]
												a3_y_global = output_switches3_yt[f3, z1, z2, img]
												
												# pool2 -> conv2
												a2_x_global = output_switches2_x[f2, a3_x_global, a3_y_global, img] + a2_x
												a2_y_global = output_switches2_y[f2, a3_x_global, a3_y_global, img] + a2_y
												
												for a1_x_ in range(s1):
													# pool1 -> conv1
													a1_x_global = output_switches1_x[f1_, a2_x_global, a2_y_global, img] + a1_x_
													for a1_y_ in range(s1):
														# pool1 -> conv1
														a1_y_global = output_switches1_y[f1_, a2_x_global, a2_y_global, img] + a1_y_
														
														for channel_ in range(3):
															# conv1 -> imgs
															#if a1_y_global >= imgs.shape[2]:
															#	print a1_y_global, imgs.shape[2]
															#	print output_switches1_y[f1_, a2_x_global, a2_y_global, img], a1_y_
															temp_F_prod_all = FL32 * imgs[channel_, a1_x_global, a1_y_global, img]
															
															# supervised term:
															grad[f1_, channel_, a1_x_, a1_y_] -= temp_F_prod_all * Y[cat,img]
															#grad_s += np.abs(temp_F_prod_all * Y_cat_sum[img])

															# unsupervised term:
															grad[f1_, channel_, a1_x_, a1_y_] +=  temp_F_prod_all * pred[cat,img]
															#grad_uns += np.abs(temp_F_prod_all * pred_cat_sum[img])
		
	#print grad_s, grad_uns
	return grad


