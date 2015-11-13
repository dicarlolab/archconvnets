#define KEYS(A, B) keys[(A)*mem_length + B]
#define MEM(A, B) mem[(A)*mem_length + B]

static PyObject *cosine_sim_expand_dkeys_cpu(PyObject *self, PyObject *args){
    PyArrayObject *keys_in, *mem_in, *numpy_buffer_temp;
	float *keys, *mem, *keys_sq_sum, *mem_sq_sum;
	
	if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &keys_in, &PyArray_Type, &mem_in)) 
		return NULL;
    
	int n_controllers = PyArray_DIM(keys_in, 0);
	int mem_length = PyArray_DIM(keys_in, 1);
	int M = PyArray_DIM(mem_in, 0); // num mem slots
	
	keys = (float *) PyArray_DATA(keys_in);
	mem = (float *) PyArray_DATA(mem_in);
	
	/////////////////////////  output buffer
	int dims[5];
	dims[0] = n_controllers;
	dims[1] = M;
	dims[2] = n_controllers;
	dims[3] = mem_length;
	numpy_buffer_temp = (PyArrayObject *) PyArray_FromDims(4, dims, NPY_FLOAT);
	
	//////////////////////// keys_sq_sum, mem_sq_sum
	MALLOC(keys_sq_sum, n_controllers * sizeof(DATA_TYPE))
	MALLOC(mem_sq_sum, M  * sizeof(DATA_TYPE))
	
	memset(keys_sq_sum, 0, n_controllers * sizeof(DATA_TYPE));
	memset(mem_sq_sum, 0, M * sizeof(DATA_TYPE));
	
	for(int i = 0; i < n_controllers; i++){
		for(int j = 0; j < mem_length; j++){
			keys_sq_sum[i] += KEYS(i,j) * KEYS(i,j);
		}
		keys_sq_sum[i] = sqrt(keys_sq_sum[i]);
	}
	
	for(int i = 0; i < M; i++){
		for(int j = 0; j < mem_length; j++){
			mem_sq_sum[i] += MEM(i,j) * MEM(i,j);
		}
		mem_sq_sum[i] = sqrt(mem_sq_sum[i]);
	}
	
	
	
	// denom
	// numer
	
	// keys = keys / keys_sq_sum
	
	// mem*denom - temp (keys*numer*mem_sq_sum)
	
	free(keys_sq_sum);
	free(mem_sq_sum);
	
	return PyArray_Return(numpy_buffer_temp);
}
