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

int data_sz[N_GPUS][N_BUFFERS];
float *data_buffers[N_GPUS][N_BUFFERS];

