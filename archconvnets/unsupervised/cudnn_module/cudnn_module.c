#include "includes.h"

#include "conv.c"
#include "conv_L1.c"
#include "conv_L2.c"
#include "conv_L3.c"

static PyMethodDef _cudnn_module[] = {
	{"conv", conv, METH_VARARGS},
	{"conv_L1", conv_L1, METH_VARARGS},
	{"conv_L2", conv_L2, METH_VARARGS},
	{"conv_L3", conv_L3, METH_VARARGS},
	{NULL, NULL}
};

void init_cudnn_module(){
	(void) Py_InitModule("_cudnn_module", _cudnn_module);
	import_array();
	
	cudnnStatus_t status;
	status = cudnnCreate(&handle);  ERR_CHECK_R
	
	//---------------------------------------
	// Create general Descriptors
	//---------------------------------------
	status = cudnnCreateTensor4dDescriptor(&srcDesc);  ERR_CHECK_R
	status = cudnnCreateTensor4dDescriptor(&destDesc);  ERR_CHECK_R
	status = cudnnCreateFilterDescriptor(&filterDesc);  ERR_CHECK_R
	status = cudnnCreateConvolutionDescriptor(&convDesc);  ERR_CHECK_R

	//---------------------------------------
	// Create L1 Descriptors
	//---------------------------------------
	status = cudnnCreateTensor4dDescriptor(&srcDescL1);  ERR_CHECK_R
	status = cudnnCreateTensor4dDescriptor(&destDescL1);  ERR_CHECK_R
	status = cudnnCreateFilterDescriptor(&filterDescL1);  ERR_CHECK_R
	status = cudnnCreateConvolutionDescriptor(&convDescL1);  ERR_CHECK_R
	
	//---------------------------------------
	// Create L2 Descriptors
	//---------------------------------------
	status = cudnnCreateTensor4dDescriptor(&srcDescL2);  ERR_CHECK_R
	status = cudnnCreateTensor4dDescriptor(&destDescL2);  ERR_CHECK_R
	status = cudnnCreateFilterDescriptor(&filterDescL2);  ERR_CHECK_R
	status = cudnnCreateConvolutionDescriptor(&convDescL2);  ERR_CHECK_R
	
	//---------------------------------------
	// Create L3 Descriptors
	//---------------------------------------
	status = cudnnCreateTensor4dDescriptor(&srcDescL3);  ERR_CHECK_R
	status = cudnnCreateTensor4dDescriptor(&destDescL3);  ERR_CHECK_R
	status = cudnnCreateFilterDescriptor(&filterDescL3);  ERR_CHECK_R
	status = cudnnCreateConvolutionDescriptor(&convDescL3);  ERR_CHECK_R
} 
