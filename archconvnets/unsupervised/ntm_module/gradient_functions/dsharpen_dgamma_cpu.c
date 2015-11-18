#define W(A, B) w[(A)*dim1 + B]
#define DSDG(A, B, C) dsdg[(A)*dim1*dim0 + (B)*dim0 + C]

static PyObject *dsharpen_dgamma_cpu(PyObject *self, PyObject *args){
	PyArrayObject *w_in, *gamma_in, *numpy_buffer_temp;
	float *w, *gamma, *dsdg;
	
	if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &w_in, &PyArray_Type, &gamma_in)) 
		return NULL;
    
	int dim0 = PyArray_DIM(w_in, 0);
	int dim1 = PyArray_DIM(w_in, 1);
	
	w = (float *) PyArray_DATA(w_in);
	gamma = (float *) PyArray_DATA(gamma_in);
	
	/////////////////////////  output buffer
	int dims[5];
	dims[0] = dim0;
	dims[1] = dim1;
	dims[2] = dim0;
	dims[3] = 1;
	numpy_buffer_temp = (PyArrayObject *) PyArray_FromDims(4, dims, NPY_FLOAT);
	dsdg = (float *) PyArray_DATA(numpy_buffer_temp);

	float *wg, *ln_w_wg, wg_sum, ln_w_wg_sum;
	MALLOC(wg, dim1*sizeof(DATA_TYPE))
	MALLOC(ln_w_wg, dim1*sizeof(DATA_TYPE))

	for(int i = 0; i < dim0; i++){
		wg_sum = 0;
		ln_w_wg_sum = 0;
		for(int j = 0; j < dim1; j++){
			wg[j] = pow(W(i,j), gamma[i]);
			ln_w_wg[j] = log(W(i,j)) * wg[j];
			
			wg_sum += wg[j];
			ln_w_wg_sum += ln_w_wg[j];
		}

		ln_w_wg_sum /= wg_sum * wg_sum;
		wg_sum = 1 / wg_sum;

		for(int j = 0; j < dim1; j++){
			DSDG(i,j,i) = ln_w_wg[j] * wg_sum - wg[j] * ln_w_wg_sum;
		}
	}

	free(wg);
	free(ln_w_wg);
	
	return PyArray_Return(numpy_buffer_temp);
}
