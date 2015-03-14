static PyObject *bp_patch_sigma31(PyObject *self, PyObject *args){
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
	int deriv_ind;
	long *output_switches3_x, *output_switches3_y;
	long *output_switches2_x, *output_switches2_y;
	long *output_switches1_x, *output_switches1_y;
	
	long *output_switches3_x_s31, *output_switches3_y_s31;
	long *output_switches2_x_s31, *output_switches2_y_s31;
	long *output_switches1_x_s31, *output_switches1_y_s31;
	float *imgs, *sigma_imgs;
	float *deriv, *pred, F_prod;
	
	float *F1, *F2, *F3, *FL;
	
	if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!O!O!O!O!O!O!O!O!iO!O!O!O!O!", 
		&PyArray_Type, &output_switches3_x_in, &PyArray_Type, &output_switches3_y_in,
		&PyArray_Type, &output_switches2_x_in, &PyArray_Type, &output_switches2_y_in,
		&PyArray_Type, &output_switches1_x_in, &PyArray_Type, &output_switches1_y_in,
		&PyArray_Type, &output_switches3_x_s31_in, &PyArray_Type, &output_switches3_y_s31_in,
		&PyArray_Type, &output_switches2_x_s31_in, &PyArray_Type, &output_switches2_y_s31_in,
		&PyArray_Type, &output_switches1_x_s31_in, &PyArray_Type, &output_switches1_y_s31_in,
		&PyArray_Type, &imgs_in, &PyArray_Type, &sigma_imgs_in, &deriv_ind, 
		&PyArray_Type, &pred_in, &PyArray_Type, &F1_in, &PyArray_Type, &F2_in, &PyArray_Type, &F3_in, &PyArray_Type, &FL_in)) 
			return NULL;

	if (NULL == output_switches3_x_in || NULL == output_switches3_y_in ||
		NULL == output_switches2_x_in || NULL == output_switches2_y_in ||
		NULL == output_switches1_x_in || NULL == output_switches1_y_in ||
		NULL == output_switches3_x_s31_in || NULL == output_switches3_y_s31_in ||
		NULL == output_switches2_x_s31_in || NULL == output_switches2_y_s31_in ||
		NULL == output_switches1_x_s31_in || NULL == output_switches1_y_s31_in ||
		NULL == F1_in || NULL == F2_in || NULL == F3_in || NULL == FL_in ||
		NULL == imgs_in || NULL == sigma_imgs_in) return NULL;

	
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
	
	int f1, channel, a1_x, a1_y, f2, a2_x, a2_y, f3, a3_x, a3_y, z1, z2, cat, img, ind;
	int a3_x_global, a3_y_global, a2_x_global, a2_y_global, a1_x_global, a1_y_global;
	
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
	
	IND_DTYPE deriv_in_ind;
	
	for(cat=0; cat < N_C; cat++){ for(f1=0; f1 < n1; f1++){ for(channel=0; channel < 3; channel++){ for(a1_x=0; a1_x < s1; a1_x++){
	for(a1_y=0; a1_y < s1; a1_y++){ for(f2=0; f2 < n2; f2++){ for(a2_x=0; a2_x < s2; a2_x++){ for(a2_y=0; a2_y < s2; a2_y++){
	for(f3=0; f3 < n3; f3++){ for(a3_x=0; a3_x < s3; a3_x++){ for(a3_y=0; a3_y < s3; a3_y++){ for(z1=0; z1 < max_output_sz3; z1++){ for(z2=0; z2 < max_output_sz3; z2++){
		
		F_prod = F1[F1_IND(f1, channel, a1_x, a1_y)] * F2[F2_IND(f2, f1, a2_x, a2_y)] * F3[F3_IND(f3, f2, a3_x, a3_y)] * FL[FL_IND(cat, f3, z1, z2)];
		if(deriv_ind == 1)
			deriv_in_ind = F1_IND(f1, channel, a1_x, a1_y);
		else if(deriv_ind == 2)
			deriv_in_ind = F2_IND(f2, f1, a2_x, a2_y);
		else if(deriv_ind == 3)
			deriv_in_ind = F3_IND(f3, f2, a3_x, a3_y);
		else if(deriv_ind == 4)
			deriv_in_ind = FL_IND(cat, f3, z1, z2);
		
		///////////////////////////////////////////////////// uns
		for(img = 0; img < N_IMGS; img++){
			// pool3 -> conv3
			a3_x_global = output_switches3_x[O3_IND(img,f3,z1,z2)] + a3_x;
			a3_y_global = output_switches3_y[O3_IND(img,f3,z1,z2)] + a3_y;
			
			// pool2 -> conv2
			a2_x_global = output_switches2_x[O2_IND(img,f2,a3_x_global,a3_y_global)] + a2_x;
			a2_y_global = output_switches2_y[O2_IND(img,f2,a3_x_global,a3_y_global)] + a2_y;
			
			a1_x_global = output_switches1_x[O1_IND(img,f1,a2_x_global,a2_y_global)] + a1_x;
			a1_y_global = output_switches1_y[O1_IND(img,f1,a2_x_global,a2_y_global)] + a1_y;
			
			deriv[deriv_in_ind] += F_prod * imgs[I_IND(img, channel,a1_x_global,a1_y_global)] * pred[cat + N_C*img];
		} // img	
		
		//////////////////////////////////////////////// sup
		// pool3 -> conv3
		a3_x_global = output_switches3_x_s31[O3_IND(cat,f3,z1,z2)] + a3_x;
		a3_y_global = output_switches3_y_s31[O3_IND(cat,f3,z1,z2)] + a3_y;
		
		// pool2 -> conv2
		a2_x_global = output_switches2_x_s31[O2_IND(cat,f2,a3_x_global,a3_y_global)] + a2_x;
		a2_y_global = output_switches2_y_s31[O2_IND(cat,f2,a3_x_global,a3_y_global)] + a2_y;
		
		a1_x_global = output_switches1_x_s31[O1_IND(cat,f1,a2_x_global,a2_y_global)] + a1_x;
		a1_y_global = output_switches1_y_s31[O1_IND(cat,f1,a2_x_global,a2_y_global)] + a1_y;
		
		deriv[deriv_in_ind] -= N_IMGS * F_prod * sigma_imgs[I_IND(cat, channel,a1_x_global,a1_y_global)];

		
	}}}}}}}}}}}}} // ind

	return PyArray_Return(deriv_in);
}
