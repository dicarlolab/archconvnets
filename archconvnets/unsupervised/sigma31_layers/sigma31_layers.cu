#include "Python.h"
#include "arrayobject.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#define O3_IND(A,B,C,D)((D) + (C)*max_output_sz3 + (B)*max_output_sz3_max_output_sz3 + (A)*max_output_sz3_max_output_sz3_n3)
#define O2_IND(A,B,C,D)((D) + (C)*max_output_sz2 + (B)*max_output_sz2_max_output_sz2 + (A)*max_output_sz2_max_output_sz2_n2)
#define O1_IND(A,B,C,D)((D) + (C)*max_output_sz1 + (B)*max_output_sz1_max_output_sz1 + (A)*max_output_sz1_max_output_sz1_n1)

#define S1_IND(A,B,C,D,E)((E) + (D)*s1 + (C)*s1_s1 + (B)*s1_s1_n1 + (A)*s1_s1_n1_3)
#define S2_IND(A,B,C,D,E)((E) + (D)*s2 + (C)*s2_s2 + (B)*s2_s2_n1 + (A)*s2_s2_n1_n2)
#define S3_IND(A,B,C,D,E)((E) + (D)*s3 + (C)*s3_s3 + (B)*s3_s3_n2 + (A)*s3_s3_n2_n3)
#define SL_IND(A,B,C,D)((D) + (C)*max_output_sz3 + (B)*max_output_sz3_max_output_sz3 + (A)*max_output_sz3_max_output_sz3_n3)

#define I_IND(A,B,C,D)((D) + (C)*img_sz + (B)*img_sz_img_sz + (A)*img_sz_img_sz_3)

// output_switches3_x, output_switches3_y, [n_imgs, n3, max_output_sz3, max_output_sz3]
// output_switches2_x, output_switches2_y, [n_imgs, n2, max_output_sz2, max_output_sz2]
// output_switches1_x, output_switches1_y, [n_imgs, n1, max_output_sz1, max_output_sz1]
// ints: s1, s2, s3
// labels [n_imgs]
// imgs: [n_imgs, 3, img_sz, img_sz] (float32)
// int: N_C
static PyObject *compute_sigma31_reduced(PyObject *self, PyObject *args){
	PyArrayObject *output_switches3_x_in, *output_switches3_y_in;
	PyArrayObject *output_switches2_x_in, *output_switches2_y_in;
	PyArrayObject *output_switches1_x_in, *output_switches1_y_in;
	PyArrayObject *imgs_in, *labels_in;
	
	PyArrayObject *sigma31_L1_in, *sigma31_L2_in, *sigma31_L3_in, *sigma31_FL_in;
	PyObject * list;
	
	int dims[1];
	int s1, s2, s3, N_C;
	long *output_switches3_x, *output_switches3_y;
	long *output_switches2_x, *output_switches2_y;
	long *output_switches1_x, *output_switches1_y;
	long *labels;
	float *imgs;
	float *sigma31_L1, *sigma31_L2, *sigma31_L3, *sigma31_FL;
	
	if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!iiiO!O!i", 
		&PyArray_Type, &output_switches3_x_in, &PyArray_Type, &output_switches3_y_in,
		&PyArray_Type, &output_switches2_x_in, &PyArray_Type, &output_switches2_y_in,
		&PyArray_Type, &output_switches1_x_in, &PyArray_Type, &output_switches1_y_in,
		&s1, &s2, &s3, &PyArray_Type, &labels_in, &PyArray_Type, &imgs_in, &N_C)) 
		return NULL;

	if (NULL == output_switches3_x_in || NULL == output_switches3_y_in ||
		NULL == output_switches2_x_in || NULL == output_switches2_y_in ||
		NULL == output_switches1_x_in || NULL == output_switches1_y_in ||
		NULL == labels_in || NULL == imgs_in)  return NULL;

	imgs = (float *) imgs_in -> data;
	labels = (long *) labels_in -> data;
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
	
	dims[0] = N_C * 3 * n1 * s1 * s1;
	
	sigma31_L1_in = (PyArrayObject *) PyArray_FromDims(1, dims, NPY_FLOAT);
	sigma31_L1 = (float *) sigma31_L1_in -> data;

	dims[0] = N_C * n2 * n1 * s2 * s2;
	sigma31_L2_in = (PyArrayObject *) PyArray_FromDims(1, dims, NPY_FLOAT);
	sigma31_L2 = (float *) sigma31_L2_in -> data;
	
	dims[0] = N_C * n3 * n2 * s3 * s3;
	sigma31_L3_in = (PyArrayObject *) PyArray_FromDims(1, dims, NPY_FLOAT);
	sigma31_L3 = (float *) sigma31_L3_in -> data;
	
	dims[0] = N_C * n3 * max_output_sz3 * max_output_sz3;
	sigma31_FL_in = (PyArrayObject *) PyArray_FromDims(1, dims, NPY_FLOAT);
	sigma31_FL = (float *) sigma31_FL_in -> data;
	
	
	int a3_x, a3_y, f1, f2, f3, a2_x, a2_y, z1, z2, img, cat, a1_x, a1_y, channel;
	int a3_x_global, a3_y_global, a2_x_global, a2_y_global, a1_x_global, a1_y_global;
	float img_sum;
	
	int max_output_sz3_max_output_sz3 = max_output_sz3*max_output_sz3;
	int max_output_sz3_max_output_sz3_n3 = max_output_sz3*max_output_sz3*n3;
	
	int max_output_sz2_max_output_sz2 = max_output_sz2*max_output_sz2;
	int max_output_sz2_max_output_sz2_n2 = max_output_sz2*max_output_sz2*n2;
	
	int max_output_sz1_max_output_sz1 = max_output_sz1*max_output_sz1;
	int max_output_sz1_max_output_sz1_n1 = max_output_sz1*max_output_sz1*n1;
	
	int s1_s1 = s1*s1;
	int s1_s1_n1 = s1*s1*n1;
	int s1_s1_n1_3 = s1*s1*n1*3;
	
	int s2_s2_n1_n2 = s2*s2*n1*n2;
	int s2_s2_n1 = s2*s2*n1;
	int s2_s2 = s2*s2;
	
	int s3_s3_n2_n3 = s3*s3*n2*n3;
	int s3_s3_n2 = s3*s3*n2;
	int s3_s3 = s3*s3;
	
	int img_sz_img_sz_3 = img_sz*img_sz*3;
	int img_sz_img_sz = img_sz*img_sz;
	
	int sigma31_L3_ind, sigma31_L2_ind, sigma31_FL_ind;
	int output_switches3_ind;
	int img_ind;
	int output_switches1_xt, output_switches1_yt;
	int sigma31_L1_ind;
	
	for(a3_x = 0; a3_x < s3; a3_x++){ for(a3_y = 0; a3_y < s3; a3_y++){
		printf("%i %i\n", a3_x, a3_y);
		for(f1 = 0; f1 < n1; f1++){
			for(f2 = 0; f2 < n2; f2++){
				for(f3 = 0; f3 < n3; f3++){
					sigma31_L3_ind = S3_IND(0, f3,f2,a3_x,a3_y);
					for(a2_x = 0; a2_x < s2; a2_x++){ for(a2_y = 0; a2_y < s2; a2_y++){
						sigma31_L2_ind = S2_IND(0, f2,f1,a2_x,a2_y);
						for(z1 = 0; z1 < max_output_sz3; z1++){ for(z2 = 0; z2 < max_output_sz3; z2++){
							sigma31_FL_ind = SL_IND(0, f3,z1,z2);
							output_switches3_ind = O3_IND(0,f3,z1,z2);
							for(img = 0; img < N_IMGS; img++){
								cat = labels[img];
								
								// pool3 -> conv3
								a3_x_global = output_switches3_x[output_switches3_ind + max_output_sz3_max_output_sz3_n3*img] + a3_x;
								a3_y_global = output_switches3_y[output_switches3_ind + max_output_sz3_max_output_sz3_n3*img] + a3_y;
								
								// pool2 -> conv2
								a2_x_global = output_switches2_x[O2_IND(img,f2,a3_x_global,a3_y_global)] + a2_x;
								a2_y_global = output_switches2_y[O2_IND(img,f2,a3_x_global,a3_y_global)] + a2_y;
								
								output_switches1_xt = output_switches1_x[O1_IND(img,f1,a2_x_global,a2_y_global)];
								output_switches1_yt = output_switches1_y[O1_IND(img,f1,a2_x_global,a2_y_global)];
								
								img_sum = 0;
								a1_x_global = output_switches1_xt;
								sigma31_L1_ind = S1_IND(cat,0,f1,0,0);
								for(a1_x = 0; a1_x < s1; a1_x++){
									// pool1 -> conv1
									a1_y_global = output_switches1_yt;
									for(a1_y = 0; a1_y < s1; a1_y++){
										for(channel = 0; channel < 3; channel++){
											img_ind = I_IND(img,channel,a1_x_global,a1_y_global);
											sigma31_L1[sigma31_L1_ind + channel*s1_s1_n1 + a1_y] += imgs[img_ind];
											img_sum += imgs[img_ind];
										}
										a1_y_global ++;
									} // a1_y
									sigma31_L1_ind += s1;
									a1_x_global ++;
								} // a1_x
								sigma31_L2[sigma31_L2_ind + s2_s2_n1_n2*cat] += img_sum;
								sigma31_L3[sigma31_L3_ind + s3_s3_n2_n3*cat] += img_sum;
								sigma31_FL[sigma31_FL_ind + max_output_sz3_max_output_sz3_n3*cat] += img_sum;
							} // img
						}} // z1, z2
					}} // a2_x, a2_y
				} // f3
			} // f2
		} // f1
	}} // a3_x, a3_y
	
	
	list = PyList_New(4);
	if(NULL == list) return NULL;
	
	if(-1 == PyList_SetItem(list, 0, PyArray_Return(sigma31_L1_in))) return NULL;
	if(-1 == PyList_SetItem(list, 1, PyArray_Return(sigma31_L2_in))) return NULL;
	if(-1 == PyList_SetItem(list, 2, PyArray_Return(sigma31_L3_in))) return NULL;
	if(-1 == PyList_SetItem(list, 3, PyArray_Return(sigma31_FL_in))) return NULL;
	
	return list;
}

//N_C * 3 * n1 * s1 * s1 * n2 * s2 * s2 * n3 * s3 * s3 * max_output_sz3 * max_output_sz3
#define S_IND(A,B,C,D,E,F,G,H,I,J,K,L,M)((M) + (L)*max_output_sz3 + (K)*max_output_sz3_max_output_sz3 + (J)*max_output_sz3_max_output_sz3_s3 + (I)*max_output_sz3_max_output_sz3_s3_s3 + \
	(H)*max_output_sz3_max_output_sz3_s3_s3_n3 + (G)*max_output_sz3_max_output_sz3_s3_s3_n3_s2 + (F)*max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2 + (E)*max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2 + \
	(D)*max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1 + (C)*max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1 + (B)*max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1_n1 + \
	(A)*max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1_n1_3)

// output_switches3_x, output_switches3_y, [n_imgs, n3, max_output_sz3, max_output_sz3]
// output_switches2_x, output_switches2_y, [n_imgs, n2, max_output_sz2, max_output_sz2]
// output_switches1_x, output_switches1_y, [n_imgs, n1, max_output_sz1, max_output_sz1]
// ints: s1, s2, s3
// labels [n_imgs]
// imgs: [n_imgs, 3, img_sz, img_sz] (float32)
// int: N_C
static PyObject *compute_sigma31_full(PyObject *self, PyObject *args){
	PyArrayObject *output_switches3_x_in, *output_switches3_y_in;
	PyArrayObject *output_switches2_x_in, *output_switches2_y_in;
	PyArrayObject *output_switches1_x_in, *output_switches1_y_in;
	PyArrayObject *imgs_in, *labels_in;
	
	PyArrayObject *sigma31_in;
	
	int dims[1];
	int s1, s2, s3, N_C;
	long *output_switches3_x, *output_switches3_y;
	long *output_switches2_x, *output_switches2_y;
	long *output_switches1_x, *output_switches1_y;
	long *labels;
	float *imgs;
	float *sigma31;
	
	if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!iiiO!O!i", 
		&PyArray_Type, &output_switches3_x_in, &PyArray_Type, &output_switches3_y_in,
		&PyArray_Type, &output_switches2_x_in, &PyArray_Type, &output_switches2_y_in,
		&PyArray_Type, &output_switches1_x_in, &PyArray_Type, &output_switches1_y_in,
		&s1, &s2, &s3, &PyArray_Type, &labels_in, &PyArray_Type, &imgs_in, &N_C)) 
		return NULL;

	if (NULL == output_switches3_x_in || NULL == output_switches3_y_in ||
		NULL == output_switches2_x_in || NULL == output_switches2_y_in ||
		NULL == output_switches1_x_in || NULL == output_switches1_y_in ||
		NULL == labels_in || NULL == imgs_in)  return NULL;

	imgs = (float *) imgs_in -> data;
	labels = (long *) labels_in -> data;
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
	
	dims[0] = N_C * 3 * n1 * s1 * s1 * n2 * s2 * s2 * n3 * s3 * s3 * max_output_sz3 * max_output_sz3;
	
	sigma31_in = (PyArrayObject *) PyArray_FromDims(1, dims, NPY_FLOAT);
	sigma31 = (float *) sigma31_in -> data;

	int a3_x, a3_y, f1, f2, f3, a2_x, a2_y, z1, z2, img, cat, a1_x, a1_y, channel;
	int a3_x_global, a3_y_global, a2_x_global, a2_y_global, a1_x_global, a1_y_global;
	float img_sum;
	
	int max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1_n1_3 = max_output_sz3*max_output_sz3*s3*s3*n3*s2*s2*n2*s1*s1*n1*3;
	int max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1_n1 = max_output_sz3*max_output_sz3*s3*s3*n3*s2*s2*n2*s1*s1*n1;
	int max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1 = max_output_sz3*max_output_sz3*s3*s3*n3*s2*s2*n2*s1*s1;
	int max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1 = max_output_sz3*max_output_sz3*s3*s3*n3*s2*s2*n2*s1;
	int max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2 = max_output_sz3*max_output_sz3*s3*s3*n3*s2*s2*n2;
	int max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2 = max_output_sz3*max_output_sz3*s3*s3*n3*s2*s2;
	int max_output_sz3_max_output_sz3_s3_s3_n3_s2 = max_output_sz3*max_output_sz3*s3*s3*n3*s2;
	int max_output_sz3_max_output_sz3_s3_s3_n3 = max_output_sz3*max_output_sz3*s3*s3*n3;
	int max_output_sz3_max_output_sz3_s3_s3 = max_output_sz3*max_output_sz3*s3*s3;
	int max_output_sz3_max_output_sz3_s3 = max_output_sz3*max_output_sz3*s3;
	int max_output_sz3_max_output_sz3 = max_output_sz3*max_output_sz3;
	
	int max_output_sz3_max_output_sz3_n3 = max_output_sz3*max_output_sz3*n3;
	
	int max_output_sz2_max_output_sz2 = max_output_sz2*max_output_sz2;
	int max_output_sz2_max_output_sz2_n2 = max_output_sz2*max_output_sz2*n2;
	
	int max_output_sz1_max_output_sz1 = max_output_sz1*max_output_sz1;
	int max_output_sz1_max_output_sz1_n1 = max_output_sz1*max_output_sz1*n1;
	
	int img_sz_img_sz_3 = img_sz*img_sz*3;
	int img_sz_img_sz = img_sz*img_sz;
	
	int img_ind;
	int output_switches1_xt, output_switches1_yt;
	int output_switches3_ind;
	
	for(a3_x = 0; a3_x < s3; a3_x++){ for(a3_y = 0; a3_y < s3; a3_y++){
		printf("%i %i\n", a3_x, a3_y);
		for(f1 = 0; f1 < n1; f1++){
			for(f2 = 0; f2 < n2; f2++){
				for(f3 = 0; f3 < n3; f3++){
					for(a2_x = 0; a2_x < s2; a2_x++){ for(a2_y = 0; a2_y < s2; a2_y++){
						for(z1 = 0; z1 < max_output_sz3; z1++){ for(z2 = 0; z2 < max_output_sz3; z2++){
							output_switches3_ind = O3_IND(0,f3,z1,z2);
							for(img = 0; img < N_IMGS; img++){
								cat = labels[img];
								
								// pool3 -> conv3
								a3_x_global = output_switches3_x[output_switches3_ind + max_output_sz3_max_output_sz3_n3*img] + a3_x;
								a3_y_global = output_switches3_y[output_switches3_ind + max_output_sz3_max_output_sz3_n3*img] + a3_y;
								
								// pool2 -> conv2
								a2_x_global = output_switches2_x[O2_IND(img,f2,a3_x_global,a3_y_global)] + a2_x;
								a2_y_global = output_switches2_y[O2_IND(img,f2,a3_x_global,a3_y_global)] + a2_y;
								
								output_switches1_xt = output_switches1_x[O1_IND(img,f1,a2_x_global,a2_y_global)];
								output_switches1_yt = output_switches1_y[O1_IND(img,f1,a2_x_global,a2_y_global)];
								
								img_sum = 0;
								a1_x_global = output_switches1_xt;
								for(a1_x = 0; a1_x < s1; a1_x++){
									// pool1 -> conv1
									a1_y_global = output_switches1_yt;
									for(a1_y = 0; a1_y < s1; a1_y++){
										for(channel = 0; channel < 3; channel++){
											img_ind = I_IND(img,channel,a1_x_global,a1_y_global);
											//N_C * 3 * n1 * s1 * s1 * n2 * s2 * s2 * n3 * s3 * s3 * max_output_sz3 * max_output_sz3
											sigma31[S_IND(cat, channel, f1, a1_x, a1_y, f2, a2_x, a2_y, f3, a3_x, a3_y, z1, z2)] += imgs[img_ind];
											img_sum += imgs[img_ind];
										}
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
	
	return PyArray_Return(sigma31_in);
}

#define N_GPUS 4 // max GPUs supported (for memory allocation)
#define N_LAYERS 5 // this value shouldn't be changed---many things are hardcoded around the assumption that it's 4 (+1)

// GPU pointers, one for each GPU
float *F1s_c[N_GPUS], *F2s_c[N_GPUS], *F3s_c[N_GPUS], *FLs_c[N_GPUS];
float *sigma31s_c[N_GPUS][N_LAYERS]; // second dimension is the layer, generally not all GPUs will have all sigmas.

/////////////////////////////////// dimension inds used in the einsum function
int N_C, n1, n0, s1, n2, s2, n3, s3, max_output_sz3; // for the filters
int n1s[N_LAYERS], n0s[N_LAYERS], s1s[N_LAYERS], n2s[N_LAYERS], s2s[N_LAYERS], n3s[N_LAYERS], s3s[N_LAYERS], max_output_sz3s[N_LAYERS]; // for each layer's sigma

int max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1_n0_n1s[N_LAYERS], max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1_n0s[N_LAYERS], max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1s[N_LAYERS],
	max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1s[N_LAYERS], max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2s[N_LAYERS], max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2s[N_LAYERS],
	max_output_sz3_max_output_sz3_s3_s3_n3_s2s[N_LAYERS], max_output_sz3_max_output_sz3_s3_s3_n3s[N_LAYERS], max_output_sz3_max_output_sz3_s3_s3s[N_LAYERS], 
	max_output_sz3_max_output_sz3_s3s[N_LAYERS], max_output_sz3_max_output_sz3s[N_LAYERS], z2b[N_LAYERS];

float * sum_res_c[N_GPUS][N_LAYERS][2]; // output of summations, last dimension specifies whether this is the derivative or prediction for a given layer

#define MALLOC_ERR_CHECK {if (err != cudaSuccess){printf("malloc err line: %i\n",__LINE__); return NULL;}}
#define DATA_TYPE_SZ sizeof(float)

#include "einsum_deriv_gpu.cu"
#include "set_sigma_buffer.cu"
#include "set_filter_buffers.cu"
#include "einsum_return.cu"

static PyMethodDef _sigma31_layers[] = {
	{"compute_sigma31_reduced", compute_sigma31_reduced, METH_VARARGS},
	{"compute_sigma31_full", compute_sigma31_full, METH_VARARGS},
	{"einsum_deriv_gpu", einsum_deriv_gpu, METH_VARARGS},
	{"set_sigma_buffer", set_sigma_buffer, METH_VARARGS},
	{"set_filter_buffers", set_filter_buffers, METH_VARARGS},
	{"einsum_return", einsum_return, METH_VARARGS},
	{NULL, NULL}
};

extern "C" void init_sigma31_layers(){
	(void) Py_InitModule("_sigma31_layers", _sigma31_layers);
	import_array();
	
	for(int gpu = 0; gpu < N_GPUS; gpu++){
		for(int layer = 0; layer < N_LAYERS; layer++){
			sigma31s_c[gpu][layer] = 0;
		}
		F1s_c[gpu] = 0;
		F2s_c[gpu] = 0;
		F3s_c[gpu] = 0;
		FLs_c[gpu] = 0;
	}
	
	return;
} 
