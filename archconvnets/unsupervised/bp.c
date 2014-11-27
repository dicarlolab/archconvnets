// compile with: gcc bp.c -o bp -std=c99 -O3 -lm
// derivations in archconvnets/unsupervised/docs/darren_scan.pdf
// all matrices stored linearly in memory (column-major order);
// the [X]_IND(A, B, C, D) macros convert subscript indices into the linear index

//#define DEBUG 1 // undefining this removes several index bounds checks, speeding up code execution

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>

#define N_IMGS 128
#define IMG_SZ 128
#define IMG_SZ2 (IMG_SZ*IMG_SZ)

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
#define NF 16
#define n3 NF
#define n2 NF
#define n1 NF

// number of categories
#define N_C 999

#define PANIC(MSG){printf("%s, at line %i\n", MSG, __LINE__); exit(-1);}
#define MALLOC(A, SZ, SZ2){A = malloc((SZ)*(SZ2)); if (A == NULL){PANIC("mem allocation error");}}
#define CALLOC(A, SZ, SZ2){A = calloc((SZ)*(SZ2)); if (A == NULL){PANIC("mem allocation error");}}
#define MALLOC_RAND(A, SZ, SZ2){ MALLOC(A,SZ,SZ2); for(rand_ind=0; rand_ind < SZ; rand_ind++){A[rand_ind] = (float)rand()/(RAND_MAX);A[rand_ind] -= 0.5;}}

#define O_IND(A, B, C, D)((A) + (B)*(n_filters) + (C)*(n_filters)*(output_sz) + (D)*(n_filters)*(output_sz)*(output_sz))
#define F_IND(A, B, C, D)((A) + (B)*n_filters + (C)*n_filters*n_channels + (D)*n_filters*n_channels*filter_sz)
#define I_IND(A, B, C, D)((A) + (B)*n_channels + (C)*n_channels*img_sz + (D)*n_channels*img_sz*img_sz)
// output_sz: size of convolutional output in px
void conv(float *filters, float *conv_output, int n_filters, int n_channels, int filter_sz, int img_sz, float *imgs, int stride, int output_sz){
	int x_ind = 0; int y_ind = 0;
	memset(conv_output, 0, n_filters * output_sz * output_sz * N_IMGS * sizeof(float));
	for(int img = 0; img < N_IMGS; img ++){
	 for(int filter = 0; filter < n_filters; filter++){
          for(int channel = 0; channel < n_channels; channel ++){
	   x_ind = 0;
	   for(int x = 0; x < (img_sz - filter_sz)-1; x += stride){
	    y_ind = 0;
	    for(int y = 0; y < (img_sz - filter_sz)-1; y += stride){
	     for(int x_loc = 0; x_loc < filter_sz; x_loc ++){
	      for(int y_loc = 0; y_loc < filter_sz; y_loc ++){
		conv_output[O_IND(filter, x_ind, y_ind, img)] += 
					filters[F_IND(filter, channel, x_loc, y_loc)] * imgs[I_IND(channel, x + x_loc, y + y_loc, img)];
	      } // y_loc
	     } // x_loc
	     y_ind ++;
	    } // y
	    x_ind ++;
	   } // x
	  } // channel
	 } // filter
	} // img
	//printf("%f %f %f\n", filters[50],imgs[50177],conv_output[50177]);
	return;
}

#define C_IND(A, B, C, D)((A) + (B)*(n_filters) + (C)*(n_filters)*(conv_sz) + (D)*(n_filters)*(conv_sz)*(conv_sz))
#define M_IND(A, B, C, D)((A) + (B)*(n_filters) + (C)*(n_filters)*(output_sz) + (D)*(n_filters*output_sz*output_sz))
// conv_sz: size in pixels of convolutional output (conv_output)
// output_sz: size in pixels of max pooling output (max_output)
void max_pool(float *conv_output, float *max_output, int *switch_output, int n_filters, int conv_sz, int output_sz){
	int x_ind, y_ind, temp_max_ind=0, temp_compare_ind;
	float temp_max, temp_compare;
	for(int img = 0; img < N_IMGS; img++){
	 for(int filter = 0; filter < n_filters; filter++){
	  x_ind = 0;
	  for(int x = 0; x < (conv_sz - POOL_SZ)-1; x += POOL_STRIDE){
	   y_ind = 0;
	   for(int y = 0; y < (conv_sz - POOL_SZ)-1; y += POOL_STRIDE){
            temp_max = -99999;
	    for(int x_loc = 0; x_loc < POOL_SZ; x_loc++){
	     for(int y_loc = 0; y_loc < POOL_SZ; y_loc++){
		temp_compare_ind = C_IND(filter, x + x_loc, y + y_loc, img);
		temp_compare = conv_output[temp_compare_ind];
		if(temp_max < temp_compare){
			temp_max = temp_compare;
			temp_max_ind = temp_compare_ind;
		}
	     } // y_loc
	    } // x_loc
	    max_output[M_IND(filter, x_ind, y_ind, img)] = temp_max;
	    switch_output[M_IND(filter, x_ind, y_ind, img)] = temp_max_ind;
	    
	    y_ind ++;
	   } // y
	   x_ind ++;
	  } // x
	 } // filter
	} // img
}

float * imgs;
float * pred; // label predictions [number categories x N_IMGS]
int output_sz1, output_sz2, output_sz3; // size in px of each conv. output spatial size
int max_output_sz1, max_output_sz2, max_output_sz3; // size in px of each pooling output spatial size
float *F1, *F2, *F3, *FL; // filters
float *output1, *output2, *output3; // conv outputs
float *max_output1, *max_output2, *max_output3; // max pooling outputs
int *switch_output1, *switch_output2, *switch_output3; // max pooling switches (indices into previous conv layer)


#define SW3_IND(A, B, C, D)((A) + (B)*(n3) + (C)*(n3*max_output_sz3) + (D)*(n3*max_output_sz3*max_output_sz3))
#define SW2_IND(A, B, C, D)((A) + (B)*(n2) + (C)*(n2*max_output_sz2) + (D)*(n2*max_output_sz2*max_output_sz2))
#define SW1_IND(A, B, C, D)((A) + (B)*(n1) + (C)*(n1*max_output_sz1) + (D)*(n1*max_output_sz1*max_output_sz1))
#define IMG_IND(A, B, C, D)((A) + (B)*3 + (C)*3*IMG_SZ + (D)*3*IMG_SZ2)
// return pixel based on max switch locations (i.e., unpooling down to a single pixel)
// this function is called many, many times in gradient computation so needs to be very fast...
inline float return_px(int f1, int f2, int f3, int z1, int z2, int a3_x, int a3_y, int a2_x, int a2_y, int a1_x, int a1_y, int channel, int img){
	#ifdef DEBUG
	if(f1 >= n1 || f2 >= n2 || f3 >= n3 || z1 >= max_output_sz3 || z2 >= max_output_sz3 || a3_x >= s3 || a3_y >= s3 || 
			a2_x >= s2 || a2_y >= s2 || a1_x >= s1 || a1_y >= s1){
		printf("----------------------\n");
		printf("f1: %i (%i), f2: %i (%i), f3: %i (%i)\n", f1, n1, f2, n2, f3, n3);
		printf("z1: %i (%i), z2: %i\n", z1, max_output_sz3, z2);
		printf("a3_x: %i (%i), a3_y: %i\n", a3_x, s3, a3_y);
		PANIC("return_px() input indices out of bounds");
	}
	#endif
	
	int a3_x_global, a3_y_global, a2_x_global, a2_y_global, a1_x_global, a1_y_global;

	int ind = switch_output3[SW3_IND(f3, z1, z2, img)]; // pool3 -> conv3 index

	#ifdef DEBUG
	if((ind / (n3*output_sz3*output_sz3)) != img){
		printf("%i %i %i %i %i, %i\n",ind, (ind / n3*output_sz3*output_sz3), n3, output_sz3, img, SW3_IND(f3, z1, z2, img));
		PANIC("indexing problem in return_px(). switch_output3 img index is incorrect")
	}
	#endif
	
	// unravel conv3 index to get a3_x_global and a3_y_global (spatial location on conv3)
        // a3_x and a3_y are local positions within the map of pool2 pixels used to compute the value at this location
	int r = ind - img*n3*output_sz3*output_sz3;
	a3_y_global = r / (n3*output_sz3);
	r -= a3_y_global*(n3*output_sz3);
	a3_x_global = r / n3;
	if((r - a3_x_global*n3) != f3) PANIC("indexing problem in return_px(). switch_output3 filter index is incorrect")
	//printf("%i, %i, %i, %i, %i\n", (r - a3_x_global*n3), a3_x_global, a3_y_global, img, ind);
	//printf("(%i, %i, %i)\n", n3, output_sz3, N_IMGS);
	
	// SW2_IND(...) = pool2 index
	ind = switch_output2[SW2_IND(f2, a3_x_global + a3_x, a3_y_global + a3_y, img)]; // pool2 -> conv2 index

	#ifdef DEBUG
	if((ind / (n2*output_sz2*output_sz2)) != img)
                PANIC("indexing problem in return_px(). switch_output2 img index is incorrect")
	#endif
	
	r = ind - img*n2*output_sz2*output_sz2;
	a2_y_global = r / (n2*output_sz2);
	r -= a2_y_global*(n2*output_sz2);
	a2_x_global = r / n2;
	if((r - a2_x_global*n2) != f2) PANIC("indexing problem in return_px(). switch_output2 filter index is incorrect")
	//printf("%i %i, %i, %i\n", a2_x_global, a2_y_global, (r - a2_x_global*n2), f2);
		
	// SW1_IND(...) = pool1 index
	ind = switch_output1[SW1_IND(f1, a2_x_global + a2_x, a2_y_global + a2_y, img)]; // pool1 -> conv1 index

	#ifdef DEBUG
	if((ind / (n1*output_sz1*output_sz1)) != img)
                PANIC("indexing problem in return_px(). switch_output1 img index is incorrect")
	#endif
	
	r = ind - img*n1*output_sz1*output_sz1;
	a1_y_global = r / (n1*output_sz1);
	r -= a1_y_global*(n1*output_sz1);
	a1_x_global = r / n1;
	//printf("%i %i, %i, %i\n", a1_x_global, a1_y_global, (r - a1_x_global*n1), f1);
	#ifdef DEBUG
	if((r - a1_x_global*n1) != f1) PANIC("indexing problem in return_px(). switch_output1 filter index is incorrect")
	#endif
	
	return imgs[IMG_IND(channel, a1_x_global*STRIDE1 + a1_x, a1_y_global*STRIDE1 + a1_y, img)];
}

#define P_IND(A, B)(((A) + (B)*N_C))
#define FL_IND(A, B, C, D)((A) + (B)*N_C + (C)*N_C*n3 + (D)*N_C*n3*max_output_sz3)
// given max_output3 and FL, compute label predictions for each image
inline void compute_preds(){
	memset(pred, 0, N_C * N_IMGS * sizeof(float));
	for(int cat = 0; cat < N_C; cat++){
		for(int img = 0; img < N_IMGS; img++){
			for(int filter = 0; filter < n3; filter++){
				for(int x = 0; x < max_output_sz3; x++){
					for(int y = 0; y < max_output_sz3; y++){
						pred[P_IND(cat, img)] += FL[FL_IND(cat, filter, x, y)] * 
							max_output3[SW3_IND(filter, x, y, img)];
					}
				}
			}
		}
	}
}

int main(){
	srand(time(NULL));
	int rand_ind;
	unsigned t_start = (unsigned)time(NULL); 
	float output=0;
	float * y;
		
	output_sz1 = floor((IMG_SZ - s1) / STRIDE1);
	max_output_sz1 = floor((output_sz1 - POOL_SZ) / POOL_STRIDE);

	output_sz2 = floor((max_output_sz1 - s2));
	max_output_sz2 = floor((output_sz2 - POOL_SZ) / POOL_STRIDE);
	
	output_sz3 = floor((max_output_sz2 - s3));
	max_output_sz3 = floor((output_sz3 - POOL_SZ) / POOL_STRIDE);

	MALLOC(output1, n1 * output_sz1 * output_sz1 * N_IMGS, sizeof(float));
	MALLOC(max_output1, n1 * max_output_sz1 * max_output_sz1 * N_IMGS, sizeof(float));
	MALLOC(switch_output1, n1 * max_output_sz1 * max_output_sz1 * N_IMGS, sizeof(float));

	MALLOC(output2, n2 * output_sz2 * output_sz2 * N_IMGS, sizeof(float));
	MALLOC(max_output2, n2 * max_output_sz2 * max_output_sz2 * N_IMGS, sizeof(float));
        MALLOC(switch_output2, n2 * max_output_sz2 * max_output_sz2 * N_IMGS, sizeof(float));

	MALLOC(output3, n3 * output_sz3 * output_sz3 * N_IMGS, sizeof(float))
	MALLOC(max_output3, n3 * max_output_sz3 * max_output_sz3 * N_IMGS, sizeof(float));
        MALLOC(switch_output3, n3 * max_output_sz3 * max_output_sz3 * N_IMGS, sizeof(float));

	MALLOC(pred, N_C * N_IMGS, sizeof(float));

	MALLOC_RAND(F1, n1 * 3 * s1_2, sizeof(float));
	MALLOC_RAND(F2, n2 * n1 * s2_2, sizeof(float));
	MALLOC_RAND(F3, n3 * n2 * s3_2, sizeof(float));
	MALLOC_RAND(FL, N_C * n3 * max_output_sz3 * max_output_sz3, sizeof(float));
	MALLOC_RAND(imgs, 3 * IMG_SZ2 * N_IMGS, sizeof(float));
	MALLOC_RAND(y, N_C * N_IMGS, sizeof(float));
	
	printf("szs: %i %i, %i %i, %i %i\n", output_sz1, max_output_sz1, output_sz2, max_output_sz2, output_sz3, max_output_sz3);
	
	//////////////////////////////////////
	// compute model outputs with current filters
	conv(F1, output1, n1, 3, s1, IMG_SZ, imgs, STRIDE1, output_sz1);
	max_pool(output1, max_output1, switch_output1, n1, output_sz1, max_output_sz1);

	conv(F2, output2, n2, n1, s2, output_sz1, max_output1, 1, output_sz2);
	max_pool(output2, max_output2, switch_output2, n2, output_sz2, max_output_sz2);

	conv(F3, output3, n3, n2, s3, output_sz2, max_output2, 1, output_sz3);
	max_pool(output3, max_output3, switch_output3, n3, output_sz3, max_output_sz3);

	compute_preds();

	printf("forward pass took: %i sec\n", (unsigned)time(NULL) - t_start);
	
	///////////////////////////////////////////////
	// deriv F1

	t_start = (unsigned)time(NULL);
	
	int f1_ = 1;
	//int img = 2;
	int a1_x_ = 4, a1_y_ = 6, channel_ = 0;
	float temp_F_prod, px;
	#define F1_IND(A, B, C, D)((A) + (B)*n1 + (C)*n1*3 + (D)*n1*3*s1)
	#define F2_IND(A, B, C, D)((A) + (B)*n2 + (C)*n2*n1 + (D)*n2*n1*s2)
	#define F3_IND(A, B, C, D)((A) + (B)*n3 + (C)*n3*n2 + (D)*n3*n2*s3)
	#define Y_IND(A, B)((A) + (B)*N_C)
	for(int img=2; img < N_IMGS; img++){
	for(int f3=0; f3 < n3; f3++){
	 for(int z1=0; z1 < max_output_sz3; z1++){
	  for(int z2=0; z2 < max_output_sz3; z2++){
	
	  for(int f2=0; f2 < n2; f2++){
	   for(int a3_x=0; a3_x < s3; a3_x++){
	    for(int a3_y=0; a3_y < s3; a3_y++){
	
	    for(int a2_x=0; a2_x < s2; a2_x++){ 
	     for(int a2_y=0; a2_y < s2; a2_y++){
		px = return_px(f1_, f2, f3, z1, z2, a3_x, a3_y, a2_x, a2_y, a1_x_, a1_y_, channel_, img); // "X" in the derivations
		for(int cat = 0; cat < N_C; cat++){
			temp_F_prod = FL[FL_IND(cat, f3, z1, z2)] * F3[F3_IND(f3, f2, a3_x, a3_y)] * F2[F2_IND(f2, f1_, a2_x, a2_y)];
			
			// supervised term:
			//... sigma approximations ...
			output += temp_F_prod * y[Y_IND(cat, img)] * px;
			
			// unsupervised term:
			output +=  temp_F_prod * pred[P_IND(cat, img)] * px;
	}}}}}}}}}}
	printf("%f, F1 grad: %i sec\n", output, (unsigned)time(NULL) - t_start);
	return 0;
}

