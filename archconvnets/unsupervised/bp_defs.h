#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>

#define N_IMGS 2//64
#define IMG_SZ 128
#define IMG_SZ2 (IMG_SZ*IMG_SZ)

//gradient step sizes
#define eps_F1 0.00000001
#define eps_F2 0.0000001
#define eps_F3 0.000001
#define eps_FL 0.0000001

// filter sizes
#define STRIDE1 2
#define s3 3
#define s2 5
#define s1 7

#define s3_2 (s3*s3)
#define s2_2 (s2*s2)
#define s1_2 (s1*s1)

// pool
#define POOL_SZ 3
#define POOL_STRIDE 2

// number of filters
#define NF 2
#define n3 NF
#define n2 NF
#define n1 NF

// number of categories
#define N_C 2

#define PANIC(MSG){printf("%s, at line %i in %s\n", MSG, __LINE__, __FILE__); exit(-1);}
#define MALLOC(A, SZ, SZ2){A = malloc((SZ)*(SZ2)); if (A == NULL){PANIC("mem allocation error");}}
#define CALLOC(A, SZ, SZ2){A = calloc((SZ)*(SZ2)); if (A == NULL){PANIC("mem allocation error");}}
#define MALLOC_RAND(A, SZ, SZ2){ MALLOC(A,SZ,SZ2); for(rand_ind=0; rand_ind < SZ; rand_ind++){A[rand_ind] = -.5 + (float)rand()/(RAND_MAX);}}


/////////////////// indexing macros:

// macros for conv()
#define O_IND(A, B, C, D)((A) + (B)*(n_filters) + (C)*(n_filters)*(output_sz) + (D)*(n_filters)*(output_sz)*(output_sz)) // conv_output inds
#define F_IND(A, B, C, D)((A) + (B)*n_filters + (C)*n_filters*n_channels + (D)*n_filters*n_channels*filter_sz) // filter inds
#define I_IND(A, B, C, D)((A) + (B)*n_channels + (C)*n_channels*img_sz + (D)*n_channels*img_sz*img_sz) // img inds

// macros for max_pool():
#define C_IND(A, B, C, D)((A) + (B)*(n_filters) + (C)*(n_filters)*(conv_sz) + (D)*(n_filters)*(conv_sz)*(conv_sz)) // conv_output inds

// macros for return_px():
#define SW3_IND(A, B, C, D)((A) + (B)*(n3) + (C)*(n3*max_output_sz3) + (D)*(n3*max_output_sz3*max_output_sz3)) // switch index for pool1 (switch_output1[])
#define SW2_IND(A, B, C, D)((A) + (B)*(n2) + (C)*(n2*max_output_sz2) + (D)*(n2*max_output_sz2*max_output_sz2)) // ...
#define SW1_IND(A, B, C, D)((A) + (B)*(n1) + (C)*(n1*max_output_sz1) + (D)*(n1*max_output_sz1*max_output_sz1))
#define IMG_IND(A, B, C, D)((A) + (B)*3 + (C)*3*IMG_SZ + (D)*3*IMG_SZ2)

// macros for compute_preds() and main():
#define Y_IND(A, B)((A) + (B)*N_C) // prediction and label inds (Y[] and pred[])
#define FL_IND(A, B, C, D)((A) + (B)*N_C + (C)*N_C*n3 + (D)*N_C*n3*max_output_sz3) // classifier layer inds (FL[])

// macros for main():
#define F1_IND(A, B, C, D)((A) + (B)*n1 + (C)*n1*3 + (D)*n1*3*s1) // layer 1 filter inds (F1[])
#define F2_IND(A, B, C, D)((A) + (B)*n2 + (C)*n2*n1 + (D)*n2*n1*s2) // ...
#define F3_IND(A, B, C, D)((A) + (B)*n3 + (C)*n3*n2 + (D)*n3*n2*s3)


/////////////////////////////////////
// check for overflow and underflow index problems.... each time you use [X]_IND(), place a call to [X]_IND() to perform these checks
#ifdef DEBUG
	#define O_IND_DBG(A, B, C, D){if((A) < 0 || (B) < 0 || (C) < 0 || (D) < 0 || (A) >= n_filters || (B) >= output_sz || \
	   (C) >= output_sz || (D) >= N_IMGS){printf("%i (%i), %i (%i), %i, %i (%i)\n", (A), n_filters, (B), output_sz, (C), (D), N_IMGS);PANIC("index error")}}
	#define F_IND_DBG(A, B, C, D){if((A) < 0 || (B) < 0 || (C) < 0 || (D) < 0 || (A) >= n_filters || (B) >= n_channels || \
           (C) >= filter_sz || (D) >= filter_sz){printf("%i (%i), %i (%i), %i (%i), %i\n", (A), n_filters, (B), n_channels, (C), filter_sz, (D));PANIC("index error")}}
	#define I_IND_DBG(A, B, C, D){if((A) < 0 || (B) < 0 || (C) < 0 || (D) < 0 || (A) >= n_channels || (B) >= img_sz || \
           (C) >= img_sz || (D) >= N_IMGS){printf("%i (%i), %i (%i), %i, %i (%i)\n", (A), n_channels, (B), img_sz, (C), (D), N_IMGS);PANIC("index error")}}
	#define C_IND_DBG(A, B, C, D){if((A) < 0 || (B) < 0 || (C) < 0 || (D) < 0 || (A) >= n_filters || (B) >= conv_sz || \
           (C) >= conv_sz || (D) >= N_IMGS){printf("%i (%i), %i (%i), %i, %i (%i)\n", (A), n_filters, (B), conv_sz, (C), (D), N_IMGS);PANIC("index error")}}
	#define SW3_IND_DBG(A, B, C, D){if((A) < 0 || (B) < 0 || (C) < 0 || (D) < 0 || (A) >= n3 || (B) >= max_output_sz3 || \
       (C) >= max_output_sz3 || (D) >= N_IMGS){printf("%i (%i), %i (%i), %i, %i (%i)\n", (A), n3, (B), max_output_sz3, (C), (D), N_IMGS);PANIC("index error")}}
	#define SW2_IND_DBG(A, B, C, D){if((A) < 0 || (B) < 0 || (C) < 0 || (D) < 0 || (A) >= n2 || (B) >= max_output_sz2 || \
       (C) >= max_output_sz2 || (D) >= N_IMGS){printf("%i (%i), %i (%i), %i, %i (%i)\n", (A), n2, (B), max_output_sz2, (C), (D), N_IMGS);PANIC("index error")}}
	#define SW1_IND_DBG(A, B, C, D){if((A) < 0 || (B) < 0 || (C) < 0 || (D) < 0 || (A) >= n1 || (B) >= max_output_sz1 || \
       (C) >= max_output_sz1 || (D) >= N_IMGS){printf("%i (%i), %i (%i), %i, %i (%i)\n", (A), n1, (B), max_output_sz1, (C), (D), N_IMGS);PANIC("index error")}}
	#define IMG_IND_DBG(A, B, C, D){if((A) < 0 || (B) < 0 || (C) < 0 || (D) < 0 || (A) >= 3 || (B) >= IMG_SZ || \
       (C) >= IMG_SZ || (D) >= N_IMGS){printf("%i (%i), %i (%i), %i, %i (%i)\n", (A), 3, (B), IMG_SZ, (C), (D), N_IMGS);PANIC("index error")}}
	#define Y_IND_DBG(A, B){if((A) < 0 || (B) < 0 || (A) >= N_C || (B) >= N_IMGS){printf("%i (%i), %i (%i)\n", (A), N_C, (B), N_IMGS); \
		PANIC("index error")}}
	#define FL_IND_DBG(A, B, C, D){if((A) < 0 || (B) < 0 || (C) < 0 || (D) < 0 || (A) >= N_C || (B) >= n3 || \
       (C) >= max_output_sz3 || (D) >= max_output_sz3){printf("%i (%i), %i (%i), %i (%i), %i\n", (A), N_C, (B), n3, (C), max_output_sz3, (D));PANIC("index error")}}
	#define F1_IND_DBG(A, B, C, D){if((A) < 0 || (B) < 0 || (C) < 0 || (D) < 0 || (A) >= n1 || (B) >= 3 || \
       (C) >= s1 || (D) >= s1){printf("%i (%i), %i (%i), %i (%i), %i\n", (A), n1, (B), 3, (C), s1, (D));PANIC("index error")}}
	#define F2_IND_DBG(A, B, C, D){if((A) < 0 || (B) < 0 || (C) < 0 || (D) < 0 || (A) >= n2 || (B) >= n1 || \
       (C) >= s2 || (D) >= s2){printf("%i (%i), %i (%i), %i (%i), %i\n", (A), n2, (B), n1, (C), s2, (D));PANIC("index error")}}
	#define F3_IND_DBG(A, B, C, D){if((A) < 0 || (B) < 0 || (C) < 0 || (D) < 0 || (A) >= n3 || (B) >= n2 || \
       (C) >= s3 || (D) >= s3){printf("%i (%i), %i (%i), %i (%i), %i\n", (A), n3, (B), n2, (C), s3, (D));PANIC("index error")}}
#else
	#define O_IND_DBG(A, B, C, D){}
	#define F_IND_DBG(A, B, C, D){}
	#define I_IND_DBG(A, B, C, D){}
	#define C_IND_DBG(A, B, C, D){}
	#define SW3_IND_DBG(A, B, C, D){}
	#define SW2_IND_DBG(A, B, C, D){}
	#define SW1_IND_DBG(A, B, C, D){}
	#define IMG_IND_DBG(A, B, C, D){}
	#define Y_IND_DBG(A, B){}
	#define FL_IND_DBG(A, B, C, D){}
	#define F1_IND_DBG(A, B, C, D){}
	#define F2_IND_DBG(A, B, C, D){}
	#define F3_IND_DBG(A, B, C, D){}
#endif

