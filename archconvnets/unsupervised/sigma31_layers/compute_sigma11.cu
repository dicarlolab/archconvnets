#define S11_IND(A,B)((B) + (A)*(n_inds))
#define P_IND(A,B)((B) + (A)*(n_inds))

static PyObject *compute_sigma11(PyObject *self, PyObject *args){
	PyArrayObject *patches_in, *sigma11_in;
	
	int dims[14];
	
	float *patches, *sigma11;
	
	if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &patches_in)) return NULL;

	if (NULL == patches_in)  return NULL;
	
	int N_IMGS = PyArray_DIM(patches_in, 0);
	int n_inds = PyArray_DIM(patches_in, 1);

	patches = (float *) patches_in -> data;
	
	dims[0] = n_inds;
	dims[1] = n_inds;
	
	sigma11_in = (PyArrayObject *) PyArray_FromDims(2, dims, NPY_FLOAT);
	sigma11 = (float *) sigma11_in -> data;	
	
	int img, ind_i, ind_j;
	long p_ind_i;
	
	for(img = 0; img < N_IMGS; img++){
		for(ind_i = 0; ind_i < n_inds; ind_i++){
			p_ind_i = P_IND(img, ind_i);
			for(ind_j = ind_i; ind_j < n_inds; ind_j++){
				sigma11[S11_IND(ind_i, ind_j)] += patches[p_ind_i] * patches[P_IND(img, ind_j)];
			} // ind_i
		} // ind_j
	} // img
	
	//symmetrize
	for(ind_i = 0; ind_i < n_inds; ind_i++){
		for(ind_j = ind_i+1; ind_j < n_inds; ind_j++){
			sigma11[S11_IND(ind_j, ind_i)] = sigma11[S11_IND(ind_i, ind_j)];
		}
	}
	
	return PyArray_Return(sigma11_in);
}
