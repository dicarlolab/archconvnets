#define KEYS(A, B) keys[(A)*mem_length + B]
#define MEM(A, B) mem[(A)*mem_length + B]
#define COMB(A, B, C, D) comb[(A)*M*n_controllers*mem_length + \
	(B)*n_controllers*mem_length + (C)*mem_length + D]

static PyObject *cosine_sim_expand_dkeys_cpu(PyObject *self, PyObject *args){
    PyArrayObject *keys_in, *mem_in, *numpy_buffer_temp;
	float *keys, *mem, keys_sq_sum, *mem_sq_sum;
	float *comb;
	
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
	comb = (float *) PyArray_DATA(numpy_buffer_temp);
	
	//////////////////////// keys_sq_sum, mem_sq_sum
	MALLOC(mem_sq_sum, M  * sizeof(DATA_TYPE))
	memset(mem_sq_sum, 0, M * sizeof(DATA_TYPE));
	
	for(int j = 0; j < M; j++){
		for(int k = 0; k < mem_length; k++){
			mem_sq_sum[j] += MEM(j,k) * MEM(j,k);
		}
		mem_sq_sum[j] = sqrt(mem_sq_sum[j]);
	}
	
	/////////////////////////
	// mem*denom - temp (keys*numer*mem_sq_sum)
	float numer, denom;
	for(int i = 0; i < n_controllers; i++){ // [1]
		keys_sq_sum = 0;
		for(int k = 0; k < mem_length; k++){
			keys_sq_sum += KEYS(i,k) * KEYS(i,k);
		}
		keys_sq_sum = sqrt(keys_sq_sum);
		
		for(int j = 0; j < M; j++){
			denom = keys_sq_sum * mem_sq_sum[j];
			numer = 0;
			for(int k = 0; k < mem_length; k++){
				numer += KEYS(i,k) * MEM(j,k);
				COMB(i,j,i,k) = MEM(j,k) / denom;
			}
			numer /= keys_sq_sum * denom * denom / mem_sq_sum[j];
			
			for(int k = 0; k < mem_length; k++){ // [2]
				COMB(i,j,i,k) -= KEYS(i,k) * numer;
			}
		}
	}
	
	free(mem_sq_sum);
	
	return PyArray_Return(numpy_buffer_temp);
}
