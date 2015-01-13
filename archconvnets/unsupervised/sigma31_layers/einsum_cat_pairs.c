// inputs: sigma31, FL321
//N_C * n1 * 3 * s1 * s1 * n2 * s2 * s2 * n3 * s3 * s3 * max_output_sz3 * max_output_sz3

#define FL321_IND(A,B,C,D,E,F,G,H,I,J,K,L,M)((M) + (L)*max_output_sz3 + (K)*max_output_sz3_max_output_sz3 + (J)*max_output_sz3_max_output_sz3_s3 + (I)*max_output_sz3_max_output_sz3_s3_s3 + \
	(H)*max_output_sz3_max_output_sz3_s3_s3_n3 + (G)*max_output_sz3_max_output_sz3_s3_s3_n3_s2 + (F)*max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2 + (E)*max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2 + \
	(D)*max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1 + (C)*max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1 + (B)*max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1_n0 + \
	(A)*max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1_n0_n1)
	
#define S31_IND(A,B,C,D,E,F,G,H,I,J,K,L,M)((M) + (L)*max_output_sz3s + (K)*max_output_sz3_max_output_sz3s + (J)*max_output_sz3_max_output_sz3_s3s + (I)*max_output_sz3_max_output_sz3_s3_s3s + \
	(H)*max_output_sz3_max_output_sz3_s3_s3_n3s + (G)*max_output_sz3_max_output_sz3_s3_s3_n3_s2s + (F)*max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2s + (E)*max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2s + \
	(D)*max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1s + (C)*max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1s + (B)*max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1_n0s + \
	(A)*max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1_n0_n1s)

static PyObject *einsum_cat_pairs(PyObject *self, PyObject *args){
	PyArrayObject *sigma31_in, *FL321_in;
	
	PyArrayObject *sum_res_in;
	
	float *sigma31, *FL321, *sum_res;
	
	int dims[1];
	
	if (!PyArg_ParseTuple(args, "O!O!", 
		&PyArray_Type, &sigma31_in, &PyArray_Type, &FL321_in)) 
		return NULL;

	if (NULL == sigma31_in || NULL == FL321_in)  return NULL;

	sigma31 = (float *) sigma31_in -> data;
	FL321 = (float *) FL321_in -> data;
	
	int N_C = PyArray_DIM(sigma31_in, 0);
	int n1 = PyArray_DIM(FL321_in, 1);
	int n0 = PyArray_DIM(FL321_in, 2);
	int s1 = PyArray_DIM(FL321_in, 3);
	int n2 = PyArray_DIM(FL321_in, 5);
	int s2 = PyArray_DIM(FL321_in, 6);
	int n3 = PyArray_DIM(FL321_in, 8);
	int s3 = PyArray_DIM(FL321_in, 9);
	int max_output_sz3 = PyArray_DIM(FL321_in, 11);

	// dims for sigma
	int n1s = PyArray_DIM(sigma31_in, 1);
	int n0s = PyArray_DIM(sigma31_in, 2);
	int s1s = PyArray_DIM(sigma31_in, 3);
	int n2s = PyArray_DIM(sigma31_in, 5);
	int s2s = PyArray_DIM(sigma31_in, 6);
	int n3s = PyArray_DIM(sigma31_in, 8);
	int s3s = PyArray_DIM(sigma31_in, 9);
	int max_output_sz3s = PyArray_DIM(sigma31_in, 11);

	dims[0] = N_C * N_C;
	
	sum_res_in = (PyArrayObject *) PyArray_FromDims(1, dims, NPY_FLOAT);
	sum_res = (float *) sum_res_in -> data;

	// indexing products
	int max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1_n0_n1 = max_output_sz3*max_output_sz3*s3*s3*n3*s2*s2*n2*s1*s1*n0*n1;
	int max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1_n0 = max_output_sz3*max_output_sz3*s3*s3*n3*s2*s2*n2*s1*s1*n0;
	int max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1 = max_output_sz3*max_output_sz3*s3*s3*n3*s2*s2*n2*s1*s1;
	int max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1 = max_output_sz3*max_output_sz3*s3*s3*n3*s2*s2*n2*s1;
	int max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2 = max_output_sz3*max_output_sz3*s3*s3*n3*s2*s2*n2;
	int max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2 = max_output_sz3*max_output_sz3*s3*s3*n3*s2*s2;
	int max_output_sz3_max_output_sz3_s3_s3_n3_s2 = max_output_sz3*max_output_sz3*s3*s3*n3*s2;
	int max_output_sz3_max_output_sz3_s3_s3_n3 = max_output_sz3*max_output_sz3*s3*s3*n3;
	int max_output_sz3_max_output_sz3_s3_s3 = max_output_sz3*max_output_sz3*s3*s3;
	int max_output_sz3_max_output_sz3_s3 = max_output_sz3*max_output_sz3*s3;
	int max_output_sz3_max_output_sz3 = max_output_sz3*max_output_sz3;
	
	int max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1_n0_n1s = max_output_sz3s*max_output_sz3s*s3s*s3s*n3s*s2s*s2s*n2s*s1s*s1s*n0s*n1s;
	int max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1_n0s = max_output_sz3s*max_output_sz3s*s3s*s3s*n3s*s2s*s2s*n2s*s1s*s1s*n0s;
	int max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1s = max_output_sz3s*max_output_sz3s*s3s*s3s*n3s*s2s*s2s*n2s*s1s*s1s;
	int max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1s = max_output_sz3s*max_output_sz3s*s3s*s3s*n3s*s2s*s2s*n2s*s1s;
	int max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2s = max_output_sz3s*max_output_sz3s*s3s*s3s*n3s*s2s*s2s*n2s;
	int max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2s = max_output_sz3s*max_output_sz3s*s3s*s3s*n3s*s2s*s2s;
	int max_output_sz3_max_output_sz3_s3_s3_n3_s2s = max_output_sz3s*max_output_sz3s*s3s*s3s*n3s*s2s;
	int max_output_sz3_max_output_sz3_s3_s3_n3s = max_output_sz3s*max_output_sz3s*s3s*s3s*n3s;
	int max_output_sz3_max_output_sz3_s3_s3s = max_output_sz3s*max_output_sz3s*s3s*s3s;
	int max_output_sz3_max_output_sz3_s3s = max_output_sz3s*max_output_sz3s*s3s;
	int max_output_sz3_max_output_sz3s = max_output_sz3s*max_output_sz3s;
	
	// indices for FL321
	int cat_i, cat_j;
	int f1, f0;
	int s1x, s1y;
	int f2;
	int s2x, s2y;
	int f3;
	int s3x, s3y;
	int z1, z2;
	
	int z2b = 1;
	// check which dims should be broadcasted
	if(n1s != n1){
		max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1_n0s = 0;
	}
	if(n0s != n0){
		max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1s = 0;
	}
	if(s1s != s1){
		max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2s = 0;
		max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1s = 0;
	}
	if(n2s != n2){
		max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2s = 0;
	}
	if(s2s != s2){
		max_output_sz3_max_output_sz3_s3_s3_n3s = 0;
		max_output_sz3_max_output_sz3_s3_s3_n3_s2s = 0;
	}
	if(s3s != s3){
		max_output_sz3_max_output_sz3s = 0;
		max_output_sz3_max_output_sz3_s3s = 0;
	}
	if(n3s != n3){
		max_output_sz3_max_output_sz3_s3_s3s = 0;
	}
	if(max_output_sz3s != max_output_sz3){
		max_output_sz3s = 0;
		z2b = 0;
	}
	
	int sum_ind;
	int FL321_ind1, FL321_ind2, FL321_ind3, FL321_ind4, FL321_ind5, FL321_ind6, FL321_ind7, FL321_ind8, FL321_ind9, FL321_ind10, FL321_ind11;
	int FL321_ind12, FL321_ind13;
	int S31_ind2, S31_ind3, S31_ind4, S31_ind5, S31_ind6, S31_ind7, S31_ind8, S31_ind9, S31_ind10, S31_ind11, S31_ind12, S31_ind13;
	
	//N_C * n1 * 3 * s1 * s1 * n2 * s2 * s2 * n3 * s3 * s3 * max_output_sz3 * max_output_sz3
	for(cat_i = 0; cat_i < N_C; cat_i++){
		FL321_ind1 = 0;
		for(cat_j = 0; cat_j < N_C; cat_j++){
			sum_ind = cat_j*N_C + cat_i;
			FL321_ind2 = FL321_ind1;
			S31_ind2 = cat_i*max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1_n0_n1s;
			for(f1 = 0; f1 < n1; f1++){
				FL321_ind3 = FL321_ind2;
				S31_ind3 = S31_ind2;
				for(f0 = 0; f0 < n0; f0++){
					FL321_ind4 = FL321_ind3;
					S31_ind4 = S31_ind3;
					for(s1x = 0; s1x < s1; s1x++){
						FL321_ind5 = FL321_ind4;
						S31_ind5 = S31_ind4;
						for(s1y = 0; s1y < s1; s1y++){
							FL321_ind6 = FL321_ind5;
							S31_ind6 = S31_ind5;
							for(f2 = 0; f2 < n2; f2++){
								FL321_ind7 = FL321_ind6;
								S31_ind7 = S31_ind6;
								for(s2x = 0; s2x < s2; s2x++){
									FL321_ind8 = FL321_ind7;
									S31_ind8 = S31_ind7;
									for(s2y = 0; s2y < s2; s2y++){
										FL321_ind9 = FL321_ind8;
										S31_ind9 = S31_ind8;
										for(f3 = 0; f3 < n3; f3++){
											FL321_ind10 = FL321_ind9;
											S31_ind10 = S31_ind9;
											for(s3x = 0; s3x < s3; s3x++){
												FL321_ind11 = FL321_ind10;
												S31_ind11 = S31_ind10;
												for(s3y = 0; s3y < s3; s3y++){
													FL321_ind12 = FL321_ind11;
													S31_ind12 = S31_ind11;
													for(z1 = 0; z1 < max_output_sz3; z1++){ 
														FL321_ind13 = FL321_ind12;
														S31_ind13 = S31_ind12;
														for(z2 = 0; z2 < max_output_sz3; z2++){
															sum_res[sum_ind] += 
																sigma31[S31_ind13] * FL321[FL321_ind13];
															FL321_ind13 ++;
															S31_ind13 += z2b;
														} // z2
														FL321_ind12 += max_output_sz3;
														S31_ind12 += max_output_sz3s;
													} // z1
													FL321_ind11 += max_output_sz3_max_output_sz3;
													S31_ind11 += max_output_sz3_max_output_sz3s;
												}
												FL321_ind10 += max_output_sz3_max_output_sz3_s3;
												S31_ind10 += max_output_sz3_max_output_sz3_s3s;
											} // s3x, s3y
											FL321_ind9 += max_output_sz3_max_output_sz3_s3_s3;
											S31_ind9 += max_output_sz3_max_output_sz3_s3_s3s;
										} // f3
										FL321_ind8 += max_output_sz3_max_output_sz3_s3_s3_n3;
										S31_ind8 += max_output_sz3_max_output_sz3_s3_s3_n3s;
									}
									FL321_ind7 += max_output_sz3_max_output_sz3_s3_s3_n3_s2;
									S31_ind7 += max_output_sz3_max_output_sz3_s3_s3_n3_s2s;
								} // s2x, s2y
								FL321_ind6 += max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2;
								S31_ind6 += max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2s;
							} // f2
							FL321_ind5 += max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2;
							S31_ind5 += max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2s;
						}
						FL321_ind4 += max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1;
						S31_ind4 += max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1s;
					} // s1x, s1y
					FL321_ind3 += max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1;
					S31_ind3 += max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1s;
				}
				FL321_ind2 += max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1_n0;
				S31_ind2 += max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1_n0s;
			} // f1, f0
			FL321_ind1 += max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1_n0_n1;
		} // cat_j
	} // cat_i
	
	return PyArray_Return(sum_res_in);
}
