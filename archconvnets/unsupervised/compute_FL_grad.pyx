cimport numpy as npd
import numpy as np

''' filters: in_channels, filter_sz, filter_sz, n_filters
    imgs: in_channels, img_sz, img_sz, n_imgs
'''
def FL_grad(npd.ndarray[npd.float64_t, ndim=4] F1, npd.ndarray[npd.float64_t, ndim=4] F2, npd.ndarray[npd.float64_t, ndim=4] F3, npd.ndarray[npd.float64_t, ndim=4] FL, npd.ndarray[npd.int_t, ndim=4] output_switches3_x, npd.ndarray[npd.int_t, ndim=4] output_switches3_y, npd.ndarray[npd.int_t, ndim=4] output_switches2_x, npd.ndarray[npd.int_t, ndim=4] output_switches2_y, npd.ndarray[npd.int_t, ndim=4] output_switches1_x, npd.ndarray[npd.int_t, ndim=4] output_switches1_y, int s1, int s2, int s3, npd.ndarray[npd.float64_t, ndim=2] pred, npd.ndarray[npd.float64_t, ndim=2] Y, npd.ndarray[npd.float64_t, ndim=4] imgs): 
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
	cdef npd.ndarray[npd.int_t, ndim=4] output_switches2_xt
	cdef npd.ndarray[npd.int_t, ndim=4] output_switches2_yt
	cdef npd.ndarray[npd.int_t, ndim=4] output_switches1_xt
	cdef npd.ndarray[npd.int_t, ndim=4] output_switches1_yt
	cdef npd.ndarray[npd.int_t, ndim=4] output_switches3_xt
	cdef npd.ndarray[npd.int_t, ndim=4] output_switches3_yt
	#cdef float grad_s
	#cdef float grad_uns
	
	for a3_x in range(s3):
		output_switches3_xt = output_switches3_x + a3_x
		for a3_y in range(s3):
			output_switches3_yt = output_switches3_y + a3_y
			for a2_x in range(s2):
				output_switches2_xt = output_switches2_x + a2_x
				for a2_y in range(s2):
					output_switches2_yt = output_switches2_y + a2_y
					for f3_ in range(n3):
						for f2 in range(n2):
							for f1 in range(n1):
								F32 = F3[f3_, f2, a3_x, a3_y] * F2[f2, f1, a2_x, a2_y]
								for a1_x in range(s1):
									output_switches1_xt = output_switches1_x + a1_x
									for a1_y in range(s1):
										output_switches1_yt = output_switches1_y + a1_y
										for channel in range(3):
											F321 = F32 * F1[f1, channel, a1_x, a1_y]
	
											for z1_ in range(max_output_sz3):
												for z2_ in range(max_output_sz3):
													for img in range(N_IMGS):
														# pool3 -> conv3
														a3_x_global = output_switches3_xt[f3_, z1_, z2_, img]
														a3_y_global = output_switches3_yt[f3_, z1_, z2_, img]
													
														# pool2 -> conv2
														a2_x_global = output_switches2_xt[f2, a3_x_global, a3_y_global, img]
														a2_y_global = output_switches2_yt[f2, a3_x_global, a3_y_global, img]
														
														# pool1 -> conv1
														a1_x_global = output_switches1_xt[f1, a2_x_global, a2_y_global, img]
														a1_y_global = output_switches1_yt[f1, a2_x_global, a2_y_global, img]
															
														# conv1 -> imgs
														temp_F_prod_all = F321 * imgs[channel, a1_x_global, a1_y_global, img]
														
														for cat_ in range(N_C):
															# supervised term:
															grad[cat_, f3_, z1_, z2_] -= temp_F_prod_all * Y[cat_,img]
															#grad_s += np.abs(temp_F_prod_all * Y_cat_sum[img])

															# unsupervised term:
															grad[cat_, f3_, z1_, z2_] +=  temp_F_prod_all * pred[cat_,img]
															#grad_uns += np.abs(temp_F_prod_all * pred_cat_sum[img])

	#print grad_s, grad_uns
	return grad


