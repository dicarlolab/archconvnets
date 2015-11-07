#include "Python.h"
#include "arrayobject.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MALLOC_ERR_CHECK {if (err != cudaSuccess){printf("malloc err line: %i\n",__LINE__); return NULL;}}

#define DATA_TYPE_SZ sizeof(float)

#define N_BUFFERS 100
#define N_GPUS 4

#define NUMPY_BUFFER numpy_buffers[gpu_ind][buffer_ind]
#define GPU_BUFFER gpu_buffers[gpu_ind][buffer_ind]

PyArrayObject *numpy_buffers[N_GPUS][N_BUFFERS];
float *gpu_buffers[N_GPUS][N_BUFFERS];

