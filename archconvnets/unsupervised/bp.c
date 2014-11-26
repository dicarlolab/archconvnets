// compile with: gcc bp.c -o bp -std=c99 -O3 -lm
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
#define F 16
#define n3 F
#define n2 F
#define n1 F

// spatial output size of pool3
#define o3_2 225 // 15^2

#define PANIC(MSG){printf("%s, at line %i\n", MSG, __LINE__); exit(-1);}
#define MALLOC(A, SZ, SZ2){A = malloc((SZ)*(SZ2)); if (A == NULL){PANIC("mem allocation error");}}
#define CALLOC(A, SZ, SZ2){A = calloc((SZ)*(SZ2)); if (A == NULL){PANIC("mem allocation error");}}
#define MALLOC_RAND(A, SZ, SZ2){ MALLOC(A,SZ,SZ2); for(rand_ind=0; rand_ind < SZ; rand_ind++){A[rand_ind] = (float)rand()/RAND_MAX;}}

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
	printf("%f %f %f\n", filters[50],imgs[50177],conv_output[50177]);
	return;
}

#define C_IND(A, B, C, D)((A) + (B)*(n_filters) + (C)*(n_filters)*(conv_sz) + (D)*(n_filters)*(conv_sz)*(conv_sz))
#define M_IND(A, B, C, D)((A) + (B)*(n_filters) + (C)*(n_filters)*(output_sz) + (D)*(n_filters*output_sz*output_sz))
// conv_sz: size in pixels of convolutional output (conv_output)
// output_sz: size in pixels of max pooling output (max_output)
void max_pool(float *conv_output, float *max_output, int *switch_output, int n_filters, int conv_sz, int output_sz){
	int x_ind, y_ind, temp_max_ind, temp_compare_ind;
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

int main(){
	int rand_ind;
	unsigned t_start = (unsigned)time(NULL); 
	float output=0, b=0.5,c=2;
	float *F1, *F2, *F3, *imgs;
	float *output1, *output2, *output3;
	float *max_output1, *max_output2, *max_output3;
	int *switch_output1, *switch_output2, *switch_output3;
		
	int output_sz1 = floor((IMG_SZ - s1) / STRIDE1);
	int max_output_sz1 = floor((output_sz1 - POOL_SZ) / POOL_STRIDE);

	int output_sz2 = floor((max_output_sz1 - s2));
	int max_output_sz2 = floor((output_sz2 - POOL_SZ) / POOL_STRIDE);
	
	int output_sz3 = floor((output_sz2 - s3));
	int max_output_sz3 = floor((output_sz3 - POOL_SZ) / POOL_STRIDE);

	MALLOC(output1, n1 * output_sz1 * output_sz1 * N_IMGS, sizeof(float));
	MALLOC(max_output1, n1 * max_output_sz1 * max_output_sz1 * N_IMGS, sizeof(float));
	MALLOC(switch_output1, n1 * max_output_sz1 * max_output_sz1 * N_IMGS, sizeof(float));

	MALLOC(output2, n2 * output_sz2 * output_sz2 * N_IMGS, sizeof(float));
	MALLOC(max_output2, n2 * max_output_sz2 * max_output_sz2 * N_IMGS, sizeof(float));
        MALLOC(switch_output2, n2 * max_output_sz2 * max_output_sz2 * N_IMGS, sizeof(float));

	MALLOC(output3, n3 * output_sz3 * output_sz3 * N_IMGS, sizeof(float))
	MALLOC(max_output3, n3 * max_output_sz3 * max_output_sz3 * N_IMGS, sizeof(float));
        MALLOC(switch_output3, n3 * max_output_sz3 * max_output_sz3 * N_IMGS, sizeof(float));

	MALLOC_RAND(F1, n1 * 3 * s1_2, sizeof(float));
	MALLOC_RAND(F2, n2 * n1 * s2_2, sizeof(float));
	MALLOC_RAND(F3, n3 * n2 * s3_2, sizeof(float));
	MALLOC_RAND(imgs, 3 * IMG_SZ2 * N_IMGS, sizeof(float));

	// conv	
	conv(F1, output1, n1, 3, s1, IMG_SZ, imgs, STRIDE1, output_sz1);
	max_pool(output1, max_output1, switch_output1, n1, output_sz1, max_output_sz1);

	conv(F2, output2, n2, n1, s2, output_sz1, max_output1, 1, output_sz2);
	max_pool(output2, max_output2, switch_output2, n2, output_sz2, max_output_sz2);

	conv(F3, output3, n3, n2, s3, output_sz2, output2, 1, output_sz3);
	max_pool(output3, max_output3, switch_output3, n3, output_sz3, max_output_sz3);

	printf("%i %i %f,,%i\n", max_output_sz1, switch_output1[4*n1 * max_output_sz1 * max_output_sz1+2], max_output1[4*n1 * max_output_sz1 * max_output_sz1+1],4*n1 * max_output_sz1 * max_output_sz1+1);
	printf("%f\n", output2[4*n2 * output_sz2 * output_sz2 + 30]);
	printf("%i %i %f\n", max_output_sz2, switch_output2[4*n2 * max_output_sz2 * max_output_sz2+2], max_output2[4*n2 * max_output_sz2 * max_output_sz2+20]);
	printf("%i %i %f\n", max_output_sz3, switch_output3[4*n3 * max_output_sz3 * max_output_sz3+2], max_output3[4*n3 * max_output_sz3 * max_output_sz3+5000]);
	// deriv F1
	for(int f3=0; f3 < n3; f3++){
	 for(int z=0; z < o3_2; z++){
	
	  for(int f2=0; f2 < n2; f2++){
	   for(int a3=0; a3 < s3_2; a3++){
	
	    for(int a2=0; a2 < s2_2; a2++){ 
 		output += b*c;
	}}}}}
	printf("%f, %i\n", output, (unsigned)time(NULL) - t_start);
	return 0;
}

