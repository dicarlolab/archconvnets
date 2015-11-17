#define W(A, B) w[(A)*dim1 + B]
#define DSDW(A, B, C, D) dsdw[(A)*dim1*dim0*dim1 + \
	(B)*dim0*dim1 + (C)*dim1 + D]
#define WG(A, B) wg[(A)*dim1 + B]

static PyObject *dsharpen_dw_cpu(PyObject *self, PyObject *args){
	PyArrayObject *w_in, *gamma_in, *numpy_buffer_temp;
	float *w, *gamma, *dsdw;
	
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
	dims[3] = dim1;
	numpy_buffer_temp = (PyArrayObject *) PyArray_FromDims(4, dims, NPY_FLOAT);
	dsdw = (float *) PyArray_DATA(numpy_buffer_temp);

	float *wg, wg_sum, wg_sum2, g_wgm1;
	MALLOC(wg, dim1*sizeof(DATA_TYPE))

	for(int i = 0; i < dim0; i++){
		wg_sum = 0;
		for(int j = 0; j < dim1; j++){
			wg[j] = pow(W(i,j), gamma[i]);
			wg_sum += wg[j];
		}
		wg_sum2 = wg_sum * wg_sum;

		for(int j = 0; j < dim1; j++){
			g_wgm1 = gamma[i] * wg[j] / (W(i,j) * wg_sum2);
			for(int k = 0; k < dim1; k++){
				if(k != j)
					DSDW(i,k,i,j) = -g_wgm1 * wg[k];
				else
					DSDW(i,k,i,j) = g_wgm1 * (wg_sum - wg[j]);
			}
		}
	}

	free(wg);
	
	return PyArray_Return(numpy_buffer_temp);
}
