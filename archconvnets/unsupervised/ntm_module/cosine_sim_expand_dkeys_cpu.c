#define KEYS(A, B) keys[(A)*mem_length + B]
#define MEM(A, B) mem[(A)*mem_length + B]
#define NUMER(A, B) numer[(A)*M + B]
#define DENOM(A, B) denom[(A)*M + B]
#define TEMP(A, B, C) temp[(A)*M*mem_length + (B)*mem_length + C]
#define COMB(A, B, C, D) comb[(A)*M*n_controllers*mem_length + \
	(B)*n_controllers*mem_length + (C)*mem_length + D]

static PyObject *cosine_sim_expand_dkeys_cpu(PyObject *self, PyObject *args){
    PyArrayObject *keys_in, *mem_in, *numpy_buffer_temp;
	float *keys, *mem, *keys_sq_sum, *mem_sq_sum, *numer, *denom;
	float *temp, *comb;
	
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
	
	///////////// denom
	MALLOC(denom, n_controllers * M * sizeof(DATA_TYPE))
	
	for(int i = 0; i < n_controllers; i++){
		for(int j = 0; j < M; j++){
			DENOM(i,j) = keys_sq_sum[i] * mem_sq_sum[j];
		}
	}
	
	//////////// numer
	MALLOC(numer, n_controllers * M * sizeof(DATA_TYPE))
	memset(numer, 0, n_controllers * M * sizeof(DATA_TYPE));
	
	// dot: keys, mem.T
	for(int i = 0; i < n_controllers; i++){
		for(int j = 0; j < M; j++){
			for(int k = 0; k < mem_length; k++){
				NUMER(i,j) += KEYS(i,k) * MEM(j,k);
			}
		}
	}
	
	
	///////// numer = numer / denom**2
	///////// denom = 1 / denom # = denom/denom**2
	for(int i = 0; i < n_controllers; i++){
		for(int j = 0; j < M; j++){
			NUMER(i,j) /= DENOM(i,j) * DENOM(i,j);
			DENOM(i,j) = 1 / DENOM(i,j);
		}
	}
	
	////// keys = keys / keys_sq_sum
	for(int i = 0; i < n_controllers; i++){
		for(int j = 0; j < mem_length; j++){
			KEYS(i,j) /= keys_sq_sum[i];
		}
	}
	
	////////////////////////////////////////
	//temp = np.einsum(keys, [1,2], numer*mem_sq_sum[np.newaxis], [1,0], [1,0,2])

	MALLOC(temp, n_controllers * M * mem_length * sizeof(DATA_TYPE))
	
	for(int i = 0; i < n_controllers; i++){ // [1]
		for(int j = 0; j < M; j++){ // [0]
			for(int k = 0; k < mem_length; k++){ // [2]
				TEMP(i,j,k) = KEYS(i,k) * NUMER(i,j) * mem_sq_sum[j];
			}
		}
	}
	
	/////////////////////////
	// mem*denom - temp (keys*numer*mem_sq_sum)
	for(int i = 0; i < n_controllers; i++){ // [1]
		for(int j = 0; j < M; j++){ // [0]
			for(int k = 0; k < mem_length; k++){ // [2]
				COMB(i,j,i,k) = (DENOM(i,j) * MEM(j,k)) - TEMP(i,j,k);
			}
		}
	}
	
	
	free(keys_sq_sum);
	free(mem_sq_sum);
	free(denom);
	free(numer);
	free(temp);
	
	return PyArray_Return(numpy_buffer_temp);
}
