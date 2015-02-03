//N_C * n1 *3 * s1 * s1 * n2 * s2 * s2 * n3 * s3 * s3 * max_output_sz3 * max_output_sz3
#define P_IND(A,B)((B) + (A)*(n_inds))

#define F1S_IND(A, B, C, D)((D) + (C)*s1 + (B)*s1*s1 + (A)*s1*s1*3)
#define F2S_IND(A, B, C, D)((D) + (C)*s2 + (B)*s2*s2 + (A)*s2*s2*n1)
#define F3S_IND(A, B, C, D)((D) + (C)*s3 + (B)*s3*s3 + (A)*s3*s3*n2)
#define FLS_IND(A, B, C, D)((D) + (C)*max_output_sz3 + (B)*max_output_sz3*max_output_sz3 + (A)*max_output_sz3*max_output_sz3*n3)

// layer_ind defines which layer to keep
static PyObject *compute_F_layer_sum_inds(PyObject *self, PyObject *args){
	PyArrayObject *F1_in, *F2_in, *F3_in, *FL_in, *inds_in;
	PyArrayObject *FL321_in, *F_sum_in;
	
	int dims[14];
	int layer_ind;
	IND_DTYPE *inds;
	float *FL321, *F_sum;
	
	if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!i",  &PyArray_Type, &FL321_in, 
		&PyArray_Type, &F1_in, &PyArray_Type, &F2_in, &PyArray_Type, &F3_in, 
		&PyArray_Type, &FL_in, 
		&PyArray_Type, &inds_in, &layer_ind)) return NULL;

	if (NULL == FL321_in ||
		NULL == F1_in || NULL == F2_in || NULL == F3_in || NULL == FL_in)  return NULL;

	inds = (IND_DTYPE *) inds_in -> data;
	FL321 = (float *) FL321_in -> data;
	
	IND_DTYPE N_C = PyArray_DIM(FL_in, 0);
	IND_DTYPE max_output_sz3 = PyArray_DIM(FL_in, 2);
	IND_DTYPE n3 = PyArray_DIM(F3_in, 0);
	IND_DTYPE n2 = PyArray_DIM(F2_in, 0);
	IND_DTYPE n1 = PyArray_DIM(F1_in, 0);
	IND_DTYPE s1 = PyArray_DIM(F1_in, 2);
	IND_DTYPE s2 = PyArray_DIM(F2_in, 2);
	IND_DTYPE s3 = PyArray_DIM(F3_in, 2);
	IND_DTYPE n_inds = PyArray_DIM(inds_in, 0);
	IND_DTYPE n0 = 3;
	
	if(layer_ind == 1){ // F1 inds
		dims[0] = n1;
		dims[1] = 3;
		dims[2] = s1;
		dims[3] = s1;
	}else if(layer_ind == 2){
		dims[0] = n2;
		dims[1] = n1;
		dims[2] = s2;
		dims[3] = s2;
	}else if(layer_ind == 3){
		dims[0] = n3;
		dims[1] = n2;
		dims[2] = s3;
		dims[3] = s3;
	}else if(layer_ind == 4){
		dims[0] = N_C;
		dims[1] = n3;
		dims[2] = max_output_sz3;
		dims[3] = max_output_sz3;
	}else{
		printf("layer index (%i) not supported\n", layer_ind);
		return NULL;
	}
	
	F_sum_in = (PyArrayObject *) PyArray_FromDims(4, dims, NPY_FLOAT);
	F_sum = (float *) F_sum_in -> data;
	
	int f1, channel, a1_x, a1_y, f2, a2_x, a2_y, f3, a3_x, a3_y, z1, z2, cat, img, ind;
	
	IND_DTYPE r;
	
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
	
	IND_DTYPE F_sum_ind;
	
	for(ind = 0; ind < n_inds; ind++){
		////////////////////////////////////////////// unravel inds
		
		f1 = inds[ind] / (3 * s1 * s1 * n2 * s2 * s2 * n3 * s3 * s3 * max_output_sz3 * max_output_sz3);
		r = inds[ind] % (3 * s1 * s1 * n2 * s2 * s2 * n3 * s3 * s3 * max_output_sz3 * max_output_sz3);
		
		channel = r / (s1 * s1 * n2 * s2 * s2 * n3 * s3 * s3 * max_output_sz3 * max_output_sz3);
		r = r % (s1 * s1 * n2 * s2 * s2 * n3 * s3 * s3 * max_output_sz3 * max_output_sz3);
		
		a1_x = r / (s1 * n2 * s2 * s2 * n3 * s3 * s3 * max_output_sz3 * max_output_sz3);
		r = r % (s1 * n2 * s2 * s2 * n3 * s3 * s3 * max_output_sz3 * max_output_sz3);
		
		a1_y = r / (n2 * s2 * s2 * n3 * s3 * s3 * max_output_sz3 * max_output_sz3);
		r = r % (n2 * s2 * s2 * n3 * s3 * s3 * max_output_sz3 * max_output_sz3);
		
		f2 = r / (s2 * s2 * n3 * s3 * s3 * max_output_sz3 * max_output_sz3);
		r = r % (s2 * s2 * n3 * s3 * s3 * max_output_sz3 * max_output_sz3);
		
		a2_x = r / (s2 * n3 * s3 * s3 * max_output_sz3 * max_output_sz3);
		r = r % (s2 * n3 * s3 * s3 * max_output_sz3 * max_output_sz3);
		
		a2_y = r / (n3 * s3 * s3 * max_output_sz3 * max_output_sz3);
		r = r % (n3 * s3 * s3 * max_output_sz3 * max_output_sz3);
		
		f3 = r / (s3 * s3 * max_output_sz3 * max_output_sz3);
		r = r % (s3 * s3 * max_output_sz3 * max_output_sz3);
		
		a3_x = r / (s3 * max_output_sz3 * max_output_sz3);
		r = r % (s3 * max_output_sz3 * max_output_sz3);
		
		a3_y = r / (max_output_sz3 * max_output_sz3);
		r = r % (max_output_sz3 * max_output_sz3);
		
		z1 = r / (max_output_sz3);
		z2 = r % (max_output_sz3);
		
		if(layer_ind == 1){
			F_sum_ind = F1S_IND(f1, channel, a1_x, a1_y);
		}else if(layer_ind == 2){
			F_sum_ind = F2S_IND(f2, f1, a2_x, a2_y);
		}else if(layer_ind == 3){
			F_sum_ind = F3S_IND(f3, f2, a3_x, a3_y);
		}
		
		for(cat = 0; cat < N_C; cat++){
			if(layer_ind == 4)
				F_sum_ind = FLS_IND(cat, f3, z1, z2);
			F_sum[F_sum_ind] += FL321[P_IND(cat, ind)];
		} // img
	} // ind
		
	return PyArray_Return(F_sum_in);
}
