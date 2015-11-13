static PyObject *cosine_sim_expand_dkeys(PyObject *self, PyObject *args){
    PyTupleObject *keys, *mem;
	
	if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &keys, &PyArray_Type, &mem)) 
		return NULL;
    
	
	
	Py_INCREF(Py_None);
	return Py_None;
}
