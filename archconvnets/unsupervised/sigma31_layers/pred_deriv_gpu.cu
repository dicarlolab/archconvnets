//cat, *imgi, *f1i, *channeli, *a1_xi, *a1_yi
#define PRED_IND(A,B,C,D)((D) + (C)*s1 + \
	(B)*s1*s1 + \
	(A)*s1*s1*3)

__global__ void kernel_pred_deriv(float * F1, float * F2, float * F3, float * FL, int n1, int n2, int n3, int s1, int s2, int s3, int max_output_sz3, long * output_switches3_x, long * output_switches3_y, long * output_switches2_x, long * output_switches2_y,
		long * output_switches1_x, long * output_switches1_y, int N_IMGS, IND_DTYPE max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1_3_n1, IND_DTYPE max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1_3, 
		IND_DTYPE max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1, IND_DTYPE max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1, IND_DTYPE max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2,
		IND_DTYPE max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2, IND_DTYPE max_output_sz3_max_output_sz3_s3_s3_n3_s2, IND_DTYPE max_output_sz3_max_output_sz3_s3_s3_n3, IND_DTYPE max_output_sz3_max_output_sz3_s3_s3, 
		IND_DTYPE max_output_sz3_max_output_sz3_s3, IND_DTYPE max_output_sz3_max_output_sz3, int max_output_sz2, int max_output_sz1, IND_DTYPE max_output_sz3_max_output_sz3_n3, IND_DTYPE max_output_sz2_max_output_sz2,
		IND_DTYPE max_output_sz2_max_output_sz2_n2, IND_DTYPE max_output_sz1_max_output_sz1, IND_DTYPE max_output_sz1_max_output_sz1_n1, IND_DTYPE img_sz_img_sz_3, IND_DTYPE img_sz_img_sz, float * sigma31, float * imgs, int img_sz, int N_C, float * pred){
	int a3_x, a3_y, f1, f2, f3, a2_x, a2_y, z1, z2, img, cat, a1_x, a1_y, channel;
	int a3_x_global, a3_y_global, a2_x_global, a2_y_global, a1_x_global, a1_y_global;
		
	int output_switches1_xt, output_switches1_yt;
	
	int *a3_xi = &a3_x, *a3_yi = &a3_y;
	int *a2_xi = &a2_x, *a2_yi = &a2_y;
	int *a1_xi = &a1_x, *a1_yi = &a1_y;
	int *f1i = &f1;
	int *f2i = &f2;
	int *f3i = &f3;
	int *z1i = &z1;
	int *z2i = &z2;
	int *channeli = &channel;
	int *imgi = &img;
	int *cati = &cat;
	
	int a3_x_sz = s3, a3_y_sz = s3;
	int a2_x_sz = s2, a2_y_sz = s2;
	int a1_x_sz = s1, a1_y_sz = s1; //**don't unravel these across threads....
	int f1_sz = n1;
	int f2_sz = n2;
	int f3_sz = n3;
	int z1_sz = max_output_sz3;
	int z2_sz = max_output_sz3;
	int img_sz_ind = N_IMGS;
	
	////////////////////////////// what is unraveled across the grid?
	int r = blockIdx.x;
	int f1c = r / n2;
	f1i = &f1c;
	f1_sz = 1;
	
	int f2c = r % n2;
	f2i = &f2c;
	f2_sz = 1;
	
	/// y dim
	
	int t = blockIdx.y;
	int f3c = t / (s2*s2);
	t = t % (s2*s2);
	f3i = &f3c;
	f3_sz = 1;
	
	int a2_xc = t / s2;
	a2_xi = &a2_xc;
	a2_x_sz = 1;
	
	int a2_yc = t % s2;
	a2_yi = &a2_yc;
	a2_y_sz = 1;
	
	///// z dim
	int t2 = threadIdx.x;
	int z2c = t2 / (max_output_sz3*s3*s3);
	t2 = t2 % (max_output_sz3*s3*s3);
	z2i = &z2c;
	z2_sz = 1;
	
	int z1c = t2 / (s3*s3);
	t2 = t2 % (s3*s3);
	z1i = &z1c;
	z1_sz = 1;
	
	int a3_xc = t2 / s3;
	a3_xi = &a3_xc;
	a3_x_sz = 1;
	
	int a3_yc = t2 % s3;
	a3_yi = &a3_yc;
	a3_y_sz = 1;
	
	/////
	/*int imgc = blockIdx.z;
	imgi = &imgc;
	img_sz_ind = 1;*/
	
	float temp_sum;
	
	for(a3_x = 0; a3_x < a3_x_sz; a3_x++){ for(a3_y = 0; a3_y < a3_y_sz; a3_y++){ /////
		for(f1 = 0; f1 < f1_sz; f1++){ ///////
			for(f2 = 0; f2 < f2_sz; f2++){ /////
				for(f3 = 0; f3 < f3_sz; f3++){ ////////
					for(a2_x = 0; a2_x < a2_x_sz; a2_x++){ for(a2_y = 0; a2_y < a2_y_sz; a2_y++){ ////////
						for(z1 = 0; z1 < z1_sz; z1++){ for(z2 = 0; z2 < z2_sz; z2++){ //// z1
							for(img = 0; img < img_sz_ind; img++){
								// pool3 -> conv3
								a3_x_global = output_switches3_x[O3_IND(*imgi,*f3i,*z1i,*z2i)] + *a3_xi;
								a3_y_global = output_switches3_y[O3_IND(*imgi,*f3i,*z1i,*z2i)] + *a3_yi;
								
								// pool2 -> conv2
								a2_x_global = output_switches2_x[O2_IND(*imgi,*f2i,a3_x_global,a3_y_global)] + *a2_xi;
								a2_y_global = output_switches2_y[O2_IND(*imgi,*f2i,a3_x_global,a3_y_global)] + *a2_yi;
								
								output_switches1_xt = output_switches1_x[O1_IND(*imgi,*f1i,a2_x_global,a2_y_global)];
								output_switches1_yt = output_switches1_y[O1_IND(*imgi,*f1i,a2_x_global,a2_y_global)];
								
								a1_x_global = output_switches1_xt;
								for(a1_x = 0; a1_x < a1_x_sz; a1_x++){
									// pool1 -> conv1
									a1_y_global = output_switches1_yt;
									for(a1_y = 0; a1_y < a1_y_sz; a1_y++){
										for(channel = 0; channel < 3; channel++){
											temp_sum = 0;
											for(cat = 0; cat < N_C; cat++){
												//N_C * 3 * n1 * s1 * s1 * n2 * s2 * s2 * n3 * s3 * s3 * max_output_sz3 * max_output_sz3
												temp_sum += imgs[I_IND(*imgi,*channeli,a1_x_global,a1_y_global)] *
													F2[F2_IND(*f2i, *f1i, *a2_xi, *a2_yi)] * F3[F3_IND(*f3i, *f2i, *a3_xi, *a3_yi)] *
													FL[FL_IND(*cati, *f3i, *z1i, *z2i)] * pred[*imgi + N_IMGS*(*cati)];
											} // cat
											atomicAdd(&sigma31[PRED_IND(*f1i, *channeli, *a1_xi, *a1_yi)], temp_sum);
										} // channel
										a1_y_global ++;
									} // a1_y
									a1_x_global ++;
								} // a1_x
							} // img
						}} // z1, z2
					}} // a2_x, a2_y
				} // f3
			} // f2
		} // f1
	}} // a3_x, a3_y
	
	return;
}


// output_switches3_x, output_switches3_y, [n_imgs, n3, max_output_sz3, max_output_sz3]
// output_switches2_x, output_switches2_y, [n_imgs, n2, max_output_sz2, max_output_sz2]
// output_switches1_x, output_switches1_y, [n_imgs, n1, max_output_sz1, max_output_sz1]
// ints: s1, s2, s3
// imgs: [n_imgs, 3, img_sz, img_sz] (float32)
// int: N_C
static PyObject *pred_deriv_gpu(PyObject *self, PyObject *args){
	cudaError_t err;
	cudaSetDevice(0); CHECK_CUDA_ERR
	PyArrayObject *output_switches3_x_in, *output_switches3_y_in;
	PyArrayObject *output_switches2_x_in, *output_switches2_y_in;
	PyArrayObject *output_switches1_x_in, *output_switches1_y_in;
	PyArrayObject *imgs_in;
	PyArrayObject *F1_in, *F2_in, *F3_in, *FL_in, *pred_in;
	
	PyArrayObject *sigma31_in;
	
	int dims[14];
	int s1, s2, s3, N_C;
	long *output_switches3_x, *output_switches3_y, *output_switches3_x_c, *output_switches3_y_c;
	long *output_switches2_x, *output_switches2_y, *output_switches2_x_c, *output_switches2_y_c;
	long *output_switches1_x, *output_switches1_y, *output_switches1_x_c, *output_switches1_y_c;
	float *imgs, *imgs_c, *pred, *pred_c;
	float *sigma31, *sigma31_c;
	float * F1, * F1_c, * F2, * F2_c, * F3, * F3_c, * FL, * FL_c;
	int layer_ind;
	
	if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!O!O!O!O!iiiO!iiO!", 
		&PyArray_Type, &F1_in, &PyArray_Type, &F2_in, &PyArray_Type, &F3_in, &PyArray_Type, &FL_in,
		&PyArray_Type, &output_switches3_x_in, &PyArray_Type, &output_switches3_y_in,
		&PyArray_Type, &output_switches2_x_in, &PyArray_Type, &output_switches2_y_in,
		&PyArray_Type, &output_switches1_x_in, &PyArray_Type, &output_switches1_y_in,
		&s1, &s2, &s3, &PyArray_Type, &imgs_in, &N_C, &layer_ind, &PyArray_Type, &pred_in)) 
		return NULL;

	if (NULL == output_switches3_x_in || NULL == output_switches3_y_in || NULL == pred_in || NULL == F1_in || NULL == F2_in ||
		NULL == output_switches2_x_in || NULL == output_switches2_y_in || NULL == F3_in || NULL == FL_in ||
		NULL == output_switches1_x_in || NULL == output_switches1_y_in || NULL == imgs_in)  return NULL;

	imgs = (float *) imgs_in -> data;
	pred = (float *) pred_in -> data;
	
	F1 = (float *) F1_in -> data;
	F2 = (float *) F2_in -> data;
	F3 = (float *) F3_in -> data;
	FL = (float *) FL_in -> data;
	
	output_switches3_x = (long *) output_switches3_x_in -> data;
	output_switches3_y = (long *) output_switches3_y_in -> data;

	output_switches2_x = (long *) output_switches2_x_in -> data;
	output_switches2_y = (long *) output_switches2_y_in -> data;

	output_switches1_x = (long *) output_switches1_x_in -> data;
	output_switches1_y = (long *) output_switches1_y_in -> data;

	int N_IMGS = PyArray_DIM(imgs_in, 0);
	int img_sz = PyArray_DIM(imgs_in, 2);
	int max_output_sz3 = PyArray_DIM(output_switches3_x_in, 2);
	int max_output_sz2 = PyArray_DIM(output_switches2_x_in, 2);
	int max_output_sz1 = PyArray_DIM(output_switches1_x_in, 2);
	int n3 = PyArray_DIM(output_switches3_x_in, 1);
	int n2 = PyArray_DIM(output_switches2_x_in, 1);
	int n1 = PyArray_DIM(output_switches1_x_in, 1);
	
	dims[0] = n1;
	dims[1] = 3;
	dims[2] = s1;
	dims[3] = s1;
	
	
	sigma31_in = (PyArrayObject *) PyArray_FromDims(4, dims, NPY_FLOAT);
	sigma31 = (float *) sigma31_in -> data;
	
	cudaMalloc((void**) &pred_c, PyArray_NBYTES(pred_in)); CHECK_CUDA_ERR
	
	cudaMalloc((void**) &F1_c, PyArray_NBYTES(F1_in)); CHECK_CUDA_ERR
	cudaMalloc((void**) &F2_c, PyArray_NBYTES(F2_in)); CHECK_CUDA_ERR
	cudaMalloc((void**) &F3_c, PyArray_NBYTES(F3_in)); CHECK_CUDA_ERR
	cudaMalloc((void**) &FL_c, PyArray_NBYTES(FL_in)); CHECK_CUDA_ERR
	
	cudaMalloc((void**) &sigma31_c, PyArray_NBYTES(sigma31_in)); CHECK_CUDA_ERR
	
	cudaMalloc((void**) &output_switches3_x_c, PyArray_NBYTES(output_switches3_x_in)); CHECK_CUDA_ERR
	cudaMalloc((void**) &output_switches3_y_c, PyArray_NBYTES(output_switches3_y_in)); CHECK_CUDA_ERR
	
	cudaMalloc((void**) &output_switches2_x_c, PyArray_NBYTES(output_switches2_x_in)); CHECK_CUDA_ERR
	cudaMalloc((void**) &output_switches2_y_c, PyArray_NBYTES(output_switches2_y_in)); CHECK_CUDA_ERR
	
	cudaMalloc((void**) &output_switches1_x_c, PyArray_NBYTES(output_switches1_x_in)); CHECK_CUDA_ERR
	cudaMalloc((void**) &output_switches1_y_c, PyArray_NBYTES(output_switches1_y_in)); CHECK_CUDA_ERR
	
	cudaMalloc((void**) &imgs_c, PyArray_NBYTES(imgs_in)); CHECK_CUDA_ERR
	
	cudaMemcpy(pred_c, pred, PyArray_NBYTES(pred_in), cudaMemcpyHostToDevice);  CHECK_CUDA_ERR
	
	cudaMemcpy(F1_c, F1, PyArray_NBYTES(F1_in), cudaMemcpyHostToDevice);  CHECK_CUDA_ERR
	cudaMemcpy(F2_c, F2, PyArray_NBYTES(F2_in), cudaMemcpyHostToDevice);  CHECK_CUDA_ERR
	cudaMemcpy(F3_c, F3, PyArray_NBYTES(F3_in), cudaMemcpyHostToDevice);  CHECK_CUDA_ERR
	cudaMemcpy(FL_c, FL, PyArray_NBYTES(FL_in), cudaMemcpyHostToDevice);  CHECK_CUDA_ERR
	
	cudaMemcpy(sigma31_c, sigma31, PyArray_NBYTES(sigma31_in), cudaMemcpyHostToDevice);  CHECK_CUDA_ERR
	
	cudaMemcpy(output_switches3_x_c, output_switches3_x, PyArray_NBYTES(output_switches3_x_in), cudaMemcpyHostToDevice);  CHECK_CUDA_ERR
	cudaMemcpy(output_switches3_y_c, output_switches3_y, PyArray_NBYTES(output_switches3_y_in), cudaMemcpyHostToDevice);  CHECK_CUDA_ERR
	
	cudaMemcpy(output_switches2_x_c, output_switches2_x, PyArray_NBYTES(output_switches2_x_in), cudaMemcpyHostToDevice);  CHECK_CUDA_ERR
	cudaMemcpy(output_switches2_y_c, output_switches2_y, PyArray_NBYTES(output_switches2_y_in), cudaMemcpyHostToDevice);  CHECK_CUDA_ERR
	
	cudaMemcpy(output_switches1_x_c, output_switches1_x, PyArray_NBYTES(output_switches1_x_in), cudaMemcpyHostToDevice);  CHECK_CUDA_ERR
	cudaMemcpy(output_switches1_y_c, output_switches1_y, PyArray_NBYTES(output_switches1_y_in), cudaMemcpyHostToDevice);  CHECK_CUDA_ERR
	
	cudaMemcpy(imgs_c, imgs, PyArray_NBYTES(imgs_in), cudaMemcpyHostToDevice);  CHECK_CUDA_ERR

	IND_DTYPE max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1_3_n1 = max_output_sz3*max_output_sz3*s3*s3*n3*s2*s2*n2*s1*s1*3*n1;
	IND_DTYPE max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1_3 = max_output_sz3*max_output_sz3*s3*s3*n3*s2*s2*n2*s1*s1*3;
	IND_DTYPE max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1 = max_output_sz3*max_output_sz3*s3*s3*n3*s2*s2*n2*s1*s1;
	IND_DTYPE max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1 = max_output_sz3*max_output_sz3*s3*s3*n3*s2*s2*n2*s1;
	IND_DTYPE max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2 = max_output_sz3*max_output_sz3*s3*s3*n3*s2*s2*n2;
	IND_DTYPE max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2 = max_output_sz3*max_output_sz3*s3*s3*n3*s2*s2;
	IND_DTYPE max_output_sz3_max_output_sz3_s3_s3_n3_s2 = max_output_sz3*max_output_sz3*s3*s3*n3*s2;
	IND_DTYPE max_output_sz3_max_output_sz3_s3_s3_n3 = max_output_sz3*max_output_sz3*s3*s3*n3;
	IND_DTYPE max_output_sz3_max_output_sz3_s3_s3 = max_output_sz3*max_output_sz3*s3*s3;
	IND_DTYPE max_output_sz3_max_output_sz3_s3 = max_output_sz3*max_output_sz3*s3;
	IND_DTYPE max_output_sz3_max_output_sz3 = max_output_sz3*max_output_sz3;
	
	IND_DTYPE max_output_sz3_max_output_sz3_n3 = max_output_sz3*max_output_sz3*n3;
	
	IND_DTYPE max_output_sz2_max_output_sz2 = max_output_sz2*max_output_sz2;
	IND_DTYPE max_output_sz2_max_output_sz2_n2 = max_output_sz2*max_output_sz2*n2;
	
	IND_DTYPE max_output_sz1_max_output_sz1 = max_output_sz1*max_output_sz1;
	IND_DTYPE max_output_sz1_max_output_sz1_n1 = max_output_sz1*max_output_sz1*n1;
	
	IND_DTYPE img_sz_img_sz_3 = img_sz*img_sz*3;
	IND_DTYPE img_sz_img_sz = img_sz*img_sz;
	
	dim3 grid_sz;
	grid_sz.x = n1*n2;
	grid_sz.y = n3*s2*s2;
	//grid_sz.z = N_IMGS;//max_output_sz3*max_output_sz3*s3*s3;
	
	kernel_pred_deriv <<< grid_sz, max_output_sz3*max_output_sz3*s3*s3 >>>(F1_c, F2_c, F3_c, FL_c,
		n1, n2, n3, s1, s2, s3, max_output_sz3, output_switches3_x_c, output_switches3_y_c, output_switches2_x_c, output_switches2_y_c,
		output_switches1_x_c, output_switches1_y_c, N_IMGS, max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1_3_n1, max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1_3, 
		max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1, max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1, max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2,
		max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2, max_output_sz3_max_output_sz3_s3_s3_n3_s2, max_output_sz3_max_output_sz3_s3_s3_n3, max_output_sz3_max_output_sz3_s3_s3, 
		max_output_sz3_max_output_sz3_s3, max_output_sz3_max_output_sz3, max_output_sz2, max_output_sz1, max_output_sz3_max_output_sz3_n3, max_output_sz2_max_output_sz2,
		max_output_sz2_max_output_sz2_n2, max_output_sz1_max_output_sz1, max_output_sz1_max_output_sz1_n1, img_sz_img_sz_3, img_sz_img_sz, sigma31_c, imgs_c, img_sz, N_C, pred_c);
	
	cudaMemcpy(sigma31, sigma31_c, PyArray_NBYTES(sigma31_in), cudaMemcpyDeviceToHost);  CHECK_CUDA_ERR
	
	cudaFree(sigma31_c);
	cudaFree(output_switches3_x_c);
	cudaFree(output_switches3_y_c);
	cudaFree(output_switches2_x_c);
	cudaFree(output_switches2_y_c);
	cudaFree(output_switches1_x_c);
	cudaFree(output_switches1_y_c);
	cudaFree(imgs_c);
	cudaFree(F1_c);
	cudaFree(F2_c);
	cudaFree(F3_c);
	cudaFree(FL_c);
	cudaFree(pred_c);
	
	return PyArray_Return(sigma31_in);
}
