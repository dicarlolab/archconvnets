#include "kernel_bp_patch_sigma31_uns.cu"
#include "kernel_bp_patch_sigma31_sup.cu"

static PyObject *bp_patch_sigma31_gpu(PyObject *self, PyObject *args){
	cudaError_t err;
	
	PyArrayObject *output_switches3_x_in, *output_switches3_y_in;
	PyArrayObject *output_switches2_x_in, *output_switches2_y_in;
	PyArrayObject *output_switches1_x_in, *output_switches1_y_in;
	
	PyArrayObject *output_switches3_x_s31_in, *output_switches3_y_s31_in;
	PyArrayObject *output_switches2_x_s31_in, *output_switches2_y_s31_in;
	PyArrayObject *output_switches1_x_s31_in, *output_switches1_y_s31_in;
	PyArrayObject *imgs_in, *sigma_imgs_in, *pred_in;
	PyArrayObject *F1_in, *F2_in, *F3_in, *FL_in;
	
	PyArrayObject *deriv_in;
	
	int dims[14];
	int deriv_ind, gpu_ind;
	long *output_switches3_x, *output_switches3_y;
	long *output_switches2_x, *output_switches2_y;
	long *output_switches1_x, *output_switches1_y;
	
	long *output_switches3_x_c, *output_switches3_y_c;
	long *output_switches2_x_c, *output_switches2_y_c;
	long *output_switches1_x_c, *output_switches1_y_c;
	
	long *output_switches3_x_s31, *output_switches3_y_s31;
	long *output_switches2_x_s31, *output_switches2_y_s31;
	long *output_switches1_x_s31, *output_switches1_y_s31;
	
	long *output_switches3_x_s31_c, *output_switches3_y_s31_c;
	long *output_switches2_x_s31_c, *output_switches2_y_s31_c;
	long *output_switches1_x_s31_c, *output_switches1_y_s31_c;
	
	float *imgs, *sigma_imgs;
	float *imgs_c, *sigma_imgs_c;
	
	float *deriv, *pred;
	float *deriv_c, *pred_c;
	
	float *F1, *F2, *F3, *FL;
	float *F1_c, *F2_c, *F3_c, *FL_c;
	
	if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!O!O!O!O!O!O!O!O!iO!O!O!O!O!i", 
		&PyArray_Type, &output_switches3_x_in, &PyArray_Type, &output_switches3_y_in,
		&PyArray_Type, &output_switches2_x_in, &PyArray_Type, &output_switches2_y_in,
		&PyArray_Type, &output_switches1_x_in, &PyArray_Type, &output_switches1_y_in,
		&PyArray_Type, &output_switches3_x_s31_in, &PyArray_Type, &output_switches3_y_s31_in,
		&PyArray_Type, &output_switches2_x_s31_in, &PyArray_Type, &output_switches2_y_s31_in,
		&PyArray_Type, &output_switches1_x_s31_in, &PyArray_Type, &output_switches1_y_s31_in,
		&PyArray_Type, &imgs_in, &PyArray_Type, &sigma_imgs_in, &deriv_ind, 
		&PyArray_Type, &pred_in, &PyArray_Type, &F1_in, &PyArray_Type, &F2_in, &PyArray_Type, &F3_in, &PyArray_Type, &FL_in, &gpu_ind)) 
			return NULL;

	if (NULL == output_switches3_x_in || NULL == output_switches3_y_in ||
		NULL == output_switches2_x_in || NULL == output_switches2_y_in ||
		NULL == output_switches1_x_in || NULL == output_switches1_y_in ||
		NULL == output_switches3_x_s31_in || NULL == output_switches3_y_s31_in ||
		NULL == output_switches2_x_s31_in || NULL == output_switches2_y_s31_in ||
		NULL == output_switches1_x_s31_in || NULL == output_switches1_y_s31_in ||
		NULL == F1_in || NULL == F2_in || NULL == F3_in || NULL == FL_in ||
		NULL == imgs_in || NULL == sigma_imgs_in) return NULL;

	if(gpu_ind < 0 || gpu_ind > N_GPUS){
		printf("invalid gpu index %i\n", gpu_ind);
		return NULL;
	}
	
	cudaSetDevice(gpu_ind); CHECK_CUDA_ERR

	
	imgs = (float *) imgs_in -> data;
	sigma_imgs = (float *) sigma_imgs_in -> data;
	pred = (float *) pred_in -> data;
	
	output_switches3_x = (long *) output_switches3_x_in -> data;
	output_switches3_y = (long *) output_switches3_y_in -> data;

	output_switches2_x = (long *) output_switches2_x_in -> data;
	output_switches2_y = (long *) output_switches2_y_in -> data;

	output_switches1_x = (long *) output_switches1_x_in -> data;
	output_switches1_y = (long *) output_switches1_y_in -> data;
	
	output_switches3_x_s31 = (long *) output_switches3_x_s31_in -> data;
	output_switches3_y_s31 = (long *) output_switches3_y_s31_in -> data;

	output_switches2_x_s31 = (long *) output_switches2_x_s31_in -> data;
	output_switches2_y_s31 = (long *) output_switches2_y_s31_in -> data;

	output_switches1_x_s31 = (long *) output_switches1_x_s31_in -> data;
	output_switches1_y_s31 = (long *) output_switches1_y_s31_in -> data;
	
	FL = (float *) FL_in -> data;
	F3 = (float *) F3_in -> data;
	F2 = (float *) F2_in -> data;
	F1 = (float *) F1_in -> data;

	IND_DTYPE N_IMGS = PyArray_DIM(imgs_in, 0);
	IND_DTYPE img_sz = PyArray_DIM(imgs_in, 2);
	IND_DTYPE max_output_sz3 = PyArray_DIM(output_switches3_x_in, 2);
	IND_DTYPE max_output_sz2 = PyArray_DIM(output_switches2_x_in, 2);
	IND_DTYPE max_output_sz1 = PyArray_DIM(output_switches1_x_in, 2);
	IND_DTYPE n3 = PyArray_DIM(output_switches3_x_in, 1);
	IND_DTYPE n2 = PyArray_DIM(output_switches2_x_in, 1);
	IND_DTYPE n1 = PyArray_DIM(output_switches1_x_in, 1);
	IND_DTYPE N_C = PyArray_DIM(sigma_imgs_in, 0);
	IND_DTYPE s1 = PyArray_DIM(F1_in, 2);
	IND_DTYPE s2 = PyArray_DIM(F2_in, 2);
	IND_DTYPE s3 = PyArray_DIM(F3_in, 2);
	
	if(deriv_ind == 1){
		dims[0] = n1;
		dims[1] = 3;
		dims[2] = s1;
		dims[3] = s1;
	}else if(deriv_ind == 2){
		dims[0] = n2;
		dims[1] = n1;
		dims[2] = s2;
		dims[3] = s2;
	}else if(deriv_ind == 3){
		dims[0] = n3;
		dims[1] = n2;
		dims[2] = s3;
		dims[3] = s3;
	}else if(deriv_ind == 4){
		dims[0] = N_C;
		dims[1] = n3;
		dims[2] = max_output_sz3;
		dims[3] = max_output_sz3;
	}else{
		printf("unsupported deriv_ind %i\n", deriv_ind);
		return NULL;
	}
	
	deriv_in = (PyArrayObject *) PyArray_FromDims(4, dims, NPY_FLOAT);
	deriv = (float *) deriv_in -> data;
	
	IND_DTYPE max_output_sz3_max_output_sz3 = max_output_sz3*max_output_sz3;
	
	IND_DTYPE max_output_sz3_max_output_sz3_n3 = max_output_sz3*max_output_sz3*n3;
	
	IND_DTYPE max_output_sz2_max_output_sz2 = max_output_sz2*max_output_sz2;
	IND_DTYPE max_output_sz2_max_output_sz2_n2 = max_output_sz2*max_output_sz2*n2;
	
	IND_DTYPE max_output_sz1_max_output_sz1 = max_output_sz1*max_output_sz1;
	IND_DTYPE max_output_sz1_max_output_sz1_n1 = max_output_sz1*max_output_sz1*n1;
	
	IND_DTYPE img_sz_img_sz_3 = img_sz*img_sz*3;
	IND_DTYPE img_sz_img_sz = img_sz*img_sz;
	
	////////////////////////////////////////////////////////////// GPU malloc
	cudaMalloc((void**) &deriv_c, PyArray_NBYTES(deriv_in)); CHECK_CUDA_ERR
	cudaMalloc((void**) &pred_c, PyArray_NBYTES(pred_in)); CHECK_CUDA_ERR
	
	cudaMalloc((void**) &F1_c, PyArray_NBYTES(F1_in)); CHECK_CUDA_ERR
	cudaMalloc((void**) &F2_c, PyArray_NBYTES(F2_in)); CHECK_CUDA_ERR
	cudaMalloc((void**) &F3_c, PyArray_NBYTES(F3_in)); CHECK_CUDA_ERR
	cudaMalloc((void**) &FL_c, PyArray_NBYTES(FL_in)); CHECK_CUDA_ERR
	
	cudaMalloc((void**) &imgs_c, PyArray_NBYTES(imgs_in)); CHECK_CUDA_ERR
	cudaMalloc((void**) &sigma_imgs_c, PyArray_NBYTES(sigma_imgs_in)); CHECK_CUDA_ERR
	
	cudaMalloc((void**) &output_switches1_x_s31_c, PyArray_NBYTES(output_switches1_x_s31_in)); CHECK_CUDA_ERR
	cudaMalloc((void**) &output_switches1_y_s31_c, PyArray_NBYTES(output_switches1_y_s31_in)); CHECK_CUDA_ERR
	cudaMalloc((void**) &output_switches2_x_s31_c, PyArray_NBYTES(output_switches2_x_s31_in)); CHECK_CUDA_ERR
	cudaMalloc((void**) &output_switches2_y_s31_c, PyArray_NBYTES(output_switches2_y_s31_in)); CHECK_CUDA_ERR
	cudaMalloc((void**) &output_switches3_x_s31_c, PyArray_NBYTES(output_switches3_x_s31_in)); CHECK_CUDA_ERR
	cudaMalloc((void**) &output_switches3_y_s31_c, PyArray_NBYTES(output_switches3_y_s31_in)); CHECK_CUDA_ERR
	
	cudaMalloc((void**) &output_switches1_x_c, PyArray_NBYTES(output_switches1_x_in)); CHECK_CUDA_ERR
	cudaMalloc((void**) &output_switches1_y_c, PyArray_NBYTES(output_switches1_y_in)); CHECK_CUDA_ERR
	cudaMalloc((void**) &output_switches2_x_c, PyArray_NBYTES(output_switches2_x_in)); CHECK_CUDA_ERR
	cudaMalloc((void**) &output_switches2_y_c, PyArray_NBYTES(output_switches2_y_in)); CHECK_CUDA_ERR
	cudaMalloc((void**) &output_switches3_x_c, PyArray_NBYTES(output_switches3_x_in)); CHECK_CUDA_ERR
	cudaMalloc((void**) &output_switches3_y_c, PyArray_NBYTES(output_switches3_y_in)); CHECK_CUDA_ERR
	
	//////////////////////////////////////////////////////////// copy to GPU
	cudaMemcpy(deriv_c, deriv, PyArray_NBYTES(deriv_in), cudaMemcpyHostToDevice);  CHECK_CUDA_ERR
	cudaMemcpy(pred_c, pred, PyArray_NBYTES(pred_in), cudaMemcpyHostToDevice);  CHECK_CUDA_ERR
	
	cudaMemcpy(F1_c, F1, PyArray_NBYTES(F1_in), cudaMemcpyHostToDevice);  CHECK_CUDA_ERR
	cudaMemcpy(F2_c, F2, PyArray_NBYTES(F2_in), cudaMemcpyHostToDevice);  CHECK_CUDA_ERR
	cudaMemcpy(F3_c, F3, PyArray_NBYTES(F3_in), cudaMemcpyHostToDevice);  CHECK_CUDA_ERR
	cudaMemcpy(FL_c, FL, PyArray_NBYTES(FL_in), cudaMemcpyHostToDevice);  CHECK_CUDA_ERR
	
	cudaMemcpy(imgs_c, imgs, PyArray_NBYTES(imgs_in), cudaMemcpyHostToDevice);  CHECK_CUDA_ERR
	cudaMemcpy(sigma_imgs_c, sigma_imgs, PyArray_NBYTES(sigma_imgs_in), cudaMemcpyHostToDevice);  CHECK_CUDA_ERR
	
	cudaMemcpy(output_switches1_x_s31_c, output_switches1_x_s31, PyArray_NBYTES(output_switches1_x_s31_in), cudaMemcpyHostToDevice);  CHECK_CUDA_ERR
	cudaMemcpy(output_switches1_y_s31_c, output_switches1_y_s31, PyArray_NBYTES(output_switches1_y_s31_in), cudaMemcpyHostToDevice);  CHECK_CUDA_ERR
	cudaMemcpy(output_switches2_x_s31_c, output_switches2_x_s31, PyArray_NBYTES(output_switches2_x_s31_in), cudaMemcpyHostToDevice);  CHECK_CUDA_ERR
	cudaMemcpy(output_switches2_y_s31_c, output_switches2_y_s31, PyArray_NBYTES(output_switches2_y_s31_in), cudaMemcpyHostToDevice);  CHECK_CUDA_ERR
	cudaMemcpy(output_switches3_x_s31_c, output_switches3_x_s31, PyArray_NBYTES(output_switches3_x_s31_in), cudaMemcpyHostToDevice);  CHECK_CUDA_ERR
	cudaMemcpy(output_switches3_y_s31_c, output_switches3_y_s31, PyArray_NBYTES(output_switches3_y_s31_in), cudaMemcpyHostToDevice);  CHECK_CUDA_ERR
	
	cudaMemcpy(output_switches1_x_c, output_switches1_x, PyArray_NBYTES(output_switches1_x_in), cudaMemcpyHostToDevice);  CHECK_CUDA_ERR
	cudaMemcpy(output_switches1_y_c, output_switches1_y, PyArray_NBYTES(output_switches1_y_in), cudaMemcpyHostToDevice);  CHECK_CUDA_ERR
	cudaMemcpy(output_switches2_x_c, output_switches2_x, PyArray_NBYTES(output_switches2_x_in), cudaMemcpyHostToDevice);  CHECK_CUDA_ERR
	cudaMemcpy(output_switches2_y_c, output_switches2_y, PyArray_NBYTES(output_switches2_y_in), cudaMemcpyHostToDevice);  CHECK_CUDA_ERR
	cudaMemcpy(output_switches3_x_c, output_switches3_x, PyArray_NBYTES(output_switches3_x_in), cudaMemcpyHostToDevice);  CHECK_CUDA_ERR
	cudaMemcpy(output_switches3_y_c, output_switches3_y, PyArray_NBYTES(output_switches3_y_in), cudaMemcpyHostToDevice);  CHECK_CUDA_ERR
	
	dim3 thread_sz;
	thread_sz.x = N_IMGS;
	thread_sz.y = s3;
	
	dim3 grid_sz;
	grid_sz.x = dims[0]*dims[1]*dims[2]*dims[3];
	//grid_sz.y = n3;
	//grid_sz.z = n2;
	
	///////////////////////////////////////////////////////////////////////
	kernel_bp_patch_sigma31_uns <<<grid_sz, thread_sz>>>(deriv_c, sigma_imgs_c, imgs_c, F1_c, F2_c, F3_c, FL_c, output_switches3_x_s31_c, output_switches3_y_s31_c, output_switches2_x_s31_c,
		output_switches2_y_s31_c, output_switches1_x_s31_c, output_switches1_y_s31_c, output_switches3_x_c, output_switches3_y_c, output_switches2_x_c, output_switches2_y_c, output_switches1_x_c, output_switches1_y_c,
		N_IMGS, N_C, 3, n1, n2, n3, s1, s2, s3, max_output_sz3, max_output_sz3_max_output_sz3,	
		max_output_sz3_max_output_sz3_n3, max_output_sz2_max_output_sz2, max_output_sz2_max_output_sz2_n2, max_output_sz1_max_output_sz1,
		max_output_sz1_max_output_sz1_n1, img_sz_img_sz_3, img_sz_img_sz, deriv_ind, max_output_sz2, max_output_sz1, pred_c, img_sz);
	
	thread_sz.x = s3;
	thread_sz.y = s3;
	kernel_bp_patch_sigma31_sup <<<dims[0]*dims[1]*dims[2]*dims[3], thread_sz>>>(deriv_c, sigma_imgs_c, imgs_c, F1_c, F2_c, F3_c, FL_c, output_switches3_x_s31_c, output_switches3_y_s31_c, output_switches2_x_s31_c,
		output_switches2_y_s31_c, output_switches1_x_s31_c, output_switches1_y_s31_c, output_switches3_x_c, output_switches3_y_c, output_switches2_x_c, output_switches2_y_c, output_switches1_x_c, output_switches1_y_c,
		N_IMGS, N_C, 3, n1, n2, n3, s1, s2, s3, max_output_sz3, max_output_sz3_max_output_sz3,	
		max_output_sz3_max_output_sz3_n3, max_output_sz2_max_output_sz2, max_output_sz2_max_output_sz2_n2, max_output_sz1_max_output_sz1,
		max_output_sz1_max_output_sz1_n1, img_sz_img_sz_3, img_sz_img_sz, deriv_ind, max_output_sz2, max_output_sz1, pred_c, img_sz);
	
	///////////////////////////////////////////////////////////////////////
	cudaMemcpy(deriv, deriv_c, PyArray_NBYTES(deriv_in), cudaMemcpyDeviceToHost);  CHECK_CUDA_ERR
	
	////////////////////////////////////////////////////////////////////////
	cudaFree(deriv_c);
	cudaFree(pred_c);
	
	cudaFree(F1_c);
	cudaFree(F2_c);
	cudaFree(F3_c);
	cudaFree(FL_c);
	
	cudaFree(imgs_c);
	cudaFree(sigma_imgs_c);
	
	cudaFree(output_switches1_x_s31_c);
	cudaFree(output_switches1_y_s31_c);
	cudaFree(output_switches2_x_s31_c);
	cudaFree(output_switches2_y_s31_c);
	cudaFree(output_switches3_x_s31_c);
	cudaFree(output_switches3_y_s31_c);
	
	cudaFree(output_switches1_x_c);
	cudaFree(output_switches1_y_c);
	cudaFree(output_switches2_x_c);
	cudaFree(output_switches2_y_c);
	cudaFree(output_switches3_x_c);
	cudaFree(output_switches3_y_c);
	
	return PyArray_Return(deriv_in);
}
