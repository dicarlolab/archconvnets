// compile with: gcc bp.c -o bp -std=c99 -O3 -lm
// derivations in archconvnets/unsupervised/docs/darren_scan.pdf
// all matrices stored linearly in memory (column-major order);
// the [X]_IND(A, B, C, D) macros convert subscript indices into the linear index

//#define DEBUG 1 // undefining this removes several index bounds checks, speeding up code execution

#include "bp_defs.h"
#include "bp_conv.c" // conv()
#include "bp_max_pool.c" // max_pool()

float * imgs;
float * pred; // label predictions [number categories x N_IMGS]
int output_sz1, output_sz2, output_sz3; // size in px of each conv. output spatial size
int max_output_sz1, max_output_sz2, max_output_sz3; // size in px of each pooling output spatial size
float *F1, *F2, *F3, *FL; // filters
float *output1, *output2, *output3; // conv outputs
float *max_output1, *max_output2, *max_output3; // max pooling outputs
int *switch_output1, *switch_output2, *switch_output3; // max pooling switches (indices into previous conv layer)
float * Y;

#include "bp_return_px.c" // return_px()
#include "bp_compute_preds_cost.c" // compute_preds(), compute_cost()

int main(){
	srand(time(NULL));
	int rand_ind;
	unsigned t_start;
	float grad;
		
	output_sz1 = floor((IMG_SZ - s1) / STRIDE1);
	max_output_sz1 = floor((output_sz1 - POOL_SZ) / POOL_STRIDE);

	output_sz2 = floor((max_output_sz1 - s2));
	max_output_sz2 = floor((output_sz2 - POOL_SZ) / POOL_STRIDE);
	
	output_sz3 = floor((max_output_sz2 - s3));
	max_output_sz3 = floor((output_sz3 - POOL_SZ) / POOL_STRIDE);

	MALLOC(output1, n1 * output_sz1 * output_sz1 * N_IMGS * 2, sizeof(float));
	MALLOC(max_output1, n1 * max_output_sz1 * max_output_sz1 * N_IMGS *2, sizeof(float));
	MALLOC(switch_output1, n1 * max_output_sz1 * max_output_sz1 * N_IMGS*2, sizeof(float));

	MALLOC(output2, n2 * output_sz2 * output_sz2 * N_IMGS*2, sizeof(float));
	MALLOC(max_output2, n2 * max_output_sz2 * max_output_sz2 * N_IMGS*2, sizeof(float));
        MALLOC(switch_output2, n2 * max_output_sz2 * max_output_sz2 * N_IMGS*2, sizeof(float));

	MALLOC(output3, n3 * output_sz3 * output_sz3 * N_IMGS*2, sizeof(float))
	MALLOC(max_output3, n3 * max_output_sz3 * max_output_sz3 * N_IMGS*2, sizeof(float));
        MALLOC(switch_output3, n3 * max_output_sz3 * max_output_sz3 * N_IMGS*2, sizeof(float));

	MALLOC(pred, N_C * N_IMGS*2, sizeof(float));

	MALLOC_RAND(F1, n1 * 3 * s1_2*2, sizeof(float));
	MALLOC_RAND(F2, n2 * n1 * s2_2*2, sizeof(float));
	MALLOC_RAND(F3, n3 * n2 * s3_2*2, sizeof(float));
	MALLOC_RAND(FL, N_C * n3 * max_output_sz3 * max_output_sz3*2, sizeof(float));
	MALLOC_RAND(imgs, 3 * IMG_SZ2 * N_IMGS*2, sizeof(float));
	MALLOC_RAND(Y, N_C * N_IMGS*2, sizeof(float));
	
	printf("szs: %i %i, %i %i, %i %i\n", output_sz1, max_output_sz1, output_sz2, max_output_sz2, output_sz3, max_output_sz3);
	
	//////////////////////////////////////
	// compute model outputs with current filters
	for(int step=0; step < 100; step++){ //gradient steps
	t_start = (unsigned)time(NULL);
	conv(F1, output1, n1, 3, s1, IMG_SZ, imgs, STRIDE1, output_sz1);
	max_pool(output1, max_output1, switch_output1, n1, output_sz1, max_output_sz1);

	conv(F2, output2, n2, n1, s2, output_sz1, max_output1, 1, output_sz2);
	max_pool(output2, max_output2, switch_output2, n2, output_sz2, max_output_sz2);

	conv(F3, output3, n3, n2, s3, output_sz2, max_output2, 1, output_sz3);
	max_pool(output3, max_output3, switch_output3, n3, output_sz3, max_output_sz3);

	compute_preds();

	printf("forward pass took: %i sec\n", (unsigned)time(NULL) - t_start);
	
	//////////////////////
	// index loop variables for gradient computations
	// variables ending in "_" are indices the gradient is taken wrt
	int f1, f2, f3, f1_ = 1, f2_ = 1, f3_ = 1;
	int cat, cat_=3;
	int a1_x, a1_y, a1_x_ = 4, a1_y_ = 6;
	int channel, channel_ = 0;
	int temp_ind;
	float temp_F_prod_all, temp_F31_prod, temp_F32_prod, temp_F321_prod, temp_F32_prod_px, temp_F3, temp_F2, temp_F21_prod;
	int z1, z2, z1_ = 0, z2_ = 1;
	int a3_x,a3_y, a3_x_,a3_y_, a2_x, a2_y, a2_x_, a2_y_, img;
	
	
	///////////////////////////////////////////////
	// deriv F1: wrt f1_, a1_x_, a1_y_, channel_
	
	t_start = (unsigned)time(NULL);
	for(f1_ = 0; f1_ < 3; f1_++){ //todo: loop over a1_x_, a1_y_, channel_
		grad = 0;
		for(f3=0; f3 < n3; f3++){
		for(f2=0; f2 < n2; f2++){
		for(a3_x=0; a3_x < s3; a3_x++){ for(a3_y=0; a3_y < s3; a3_y++){
			F3_IND_DBG(f3, f2, a3_x, a3_y)
		
			temp_F3 = F3[F3_IND(f3, f2, a3_x, a3_y)];
			
	    	for(a2_x=0; a2_x < s2; a2_x++){ for(a2_y=0; a2_y < s2; a2_y++){
			
				F2_IND_DBG(f2, f1_, a2_x, a2_y)
			
				temp_F32_prod = temp_F3 * F2[F2_IND(f2, f1_, a2_x, a2_y)];
				
				for(img=0; img < N_IMGS; img++){
				for(z1=0; z1 < max_output_sz3; z1++){ for(z2=0; z2 < max_output_sz3; z2++){ 
				
					temp_F32_prod_px = temp_F32_prod * 
						return_px(f1_, f2, f3, z1, z2, a3_x, a3_y, a2_x, a2_y, a1_x_, a1_y_, channel_, img); // return_px(): "X" in the derivations
					for(cat = 0; cat < N_C; cat++){
						FL_IND_DBG(cat, f3, z1, z2)
				
						temp_F_prod_all = FL[FL_IND(cat, f3, z1, z2)] * temp_F32_prod_px;
					
						Y_IND_DBG(cat,img);
						temp_ind = Y_IND(cat,img);
						// supervised term:
						//... sigma approximations ...
						grad -= temp_F_prod_all * Y[temp_ind];
					
						// unsupervised term:
						grad +=  temp_F_prod_all * pred[temp_ind];
		}}}}}}}}}}
		F1_IND_DBG(f1_, channel_, a1_x_, a1_y_)
			
		F1[F1_IND(f1_, channel_, a1_x_, a1_y_)] -= eps_F1*2*grad;
	}
	printf("F1 grad: %i sec\n", (unsigned)time(NULL) - t_start);
	
	///////////////////////////////////////////////
	// deriv FL: wrt cat_, f3_, z1_, z2_

	t_start = (unsigned)time(NULL);
	for(f3_=0; f3_ < 2*6; f3_++){ // todo: for loop over cat_, z1_, z2_
		grad = 0;
		for(f1=0; f1 < n1; f1++){  
		for(f2=0; f2 < n2; f2++){
		for(a3_x=0; a3_x < s3; a3_x++){ for(a3_y=0; a3_y < s3; a3_y++){
		   F3_IND_DBG(f3_,f2, a3_x, a3_y)
		   
		   temp_F3 = F3[F3_IND(f3_,f2, a3_x, a3_y)];
		   
			for(a2_x=0; a2_x < s2; a2_x++){ for(a2_y=0; a2_y < s2; a2_y++){
			
				F2_IND_DBG(f2, f1, a2_x, a2_y)
				
				temp_F32_prod = temp_F3 * F2[F2_IND(f2, f1_, a2_x, a2_y)];
				
				for(a1_x=0; a1_x < s1; a1_x++){ for(a1_y=0; a1_y < s1; a1_y++){
				
					for(channel=0; channel < 3; channel++){
						F1_IND_DBG(f1, channel, a1_x, a1_y)
						
						temp_F321_prod = temp_F32_prod * F1[F1_IND(f1, channel, a1_x, a1_y)];
						
						for(img=0; img < N_IMGS; img++){
								temp_F_prod_all = temp_F321_prod *
									return_px(f1, f2, f3_, z1_, z2_, a3_x, a3_y, a2_x, a2_y, a1_x, a1_y, channel, img); // return_px(): "X" in the derivations
								
								Y_IND_DBG(cat_,img);
								
								temp_ind = Y_IND(cat_,img);
								// supervised term:
								//... sigma approximations ...
								grad -= temp_F_prod_all * Y[temp_ind];

								// unsupervised term:
								grad +=  temp_F_prod_all * pred[temp_ind];
						}
						
		}}}}}}}}}
		FL_IND_DBG(cat_, f3_, z1_, z2_)
		FL[FL_IND(cat_, f3_, z1_, z2_)] -= eps_FL*2*grad;
	}
	printf("F1 grad: %i sec\n", (unsigned)time(NULL) - t_start);
	
	
	///////////////////////////////////////////////
	// deriv F2: wrt f2_, f1_, a2_x_, a2_y_
	
	t_start = (unsigned)time(NULL);
	for(f2_=0; f2_ < 8; f2_++){
		grad = 0;
		for(f3=0; f3 < n3; f3++){
		for(a3_x=0; a3_x < s3; a3_x++){ for(a3_y=0; a3_y < s3; a3_y++){
		   F3_IND_DBG(f3, f2_, a3_x, a3_y)
		   
		   temp_F3 = F3[F3_IND(f3,f2_, a3_x, a3_y)];
		   
		   for(a1_x=0; a1_x < s1; a1_x++){ for(a1_y=0; a1_y < s1; a1_y++){
				
			for(channel=0; channel < 3; channel++){
				F1_IND_DBG(f1_, channel, a1_x, a1_y)
					
				temp_F31_prod = temp_F3 * F1[F1_IND(f1_, channel, a1_x, a1_y)];
						
				for(img=0; img < N_IMGS; img++){
					for(z1=0; z1 < max_output_sz3; z1++){ for(z2=0; z2 < max_output_sz3; z2++){ 
				
						temp_F_prod_all = temp_F31_prod *
									return_px(f1_, f2_, f3, z1, z2, a3_x, a3_y, a2_x_, a2_y_, a1_x, a1_y, channel, img); // return_px(): "X" in the derivations
						for(cat=0; cat < N_C; cat ++){	
							Y_IND_DBG(cat,img);
						
							temp_ind = Y_IND(cat,img);
							// supervised term:
							//... sigma approximations ...
							grad -= temp_F_prod_all * Y[temp_ind];
		
							// unsupervised term:
							grad +=  temp_F_prod_all * pred[temp_ind];
						}
					}
						
		}}}}}}}}
		F2_IND_DBG(f2_, f1_, a2_x_, a2_y_)
		F2[F2_IND(f2_, f1_, a2_x_, a2_y_)] -= eps_F2*2*grad;
	}
	printf("F2 grad: %i sec\n", (unsigned)time(NULL) - t_start);
	
	
	///////////////////////////////////////////////
	// deriv F3: wrt f3_, f2_, a3_x_, a3_y_
	//?
	t_start = (unsigned)time(NULL);
	for(f3_=0; f3_ < n3; f3_++){ for(f2_=0; f2_ < 3; f2_++){
		grad = 0;
		for(a2_x=0; a2_x < s2; a2_x++){ for(a2_y=0; a2_y < s2; a2_y++){
		   F2_IND_DBG(f2_, f1, a2_x, a2_y)
		   
		   temp_F2 = F2[F2_IND(f2_, f1, a2_x, a2_y)];
		   
		   for(a1_x=0; a1_x < s1; a1_x++){ for(a1_y=0; a1_y < s1; a1_y++){
				
			for(channel=0; channel < 3; channel++){
				F1_IND_DBG(f1, channel, a1_x, a1_y)
					
				temp_F21_prod = temp_F2 * F1[F1_IND(f1, channel, a1_x, a1_y)];
						
				for(img=0; img < N_IMGS; img++){
					for(z1=0; z1 < max_output_sz3; z1++){ for(z2=0; z2 < max_output_sz3; z2++){ 
				
						temp_F_prod_all = temp_F21_prod *
									return_px(f1, f2_, f3_, z1, z2, a3_x_, a3_y_, a2_x, a2_y, a1_x, a1_y, channel, img); // return_px(): "X" in the derivations
						for(cat=0; cat < N_C; cat ++){	
							Y_IND_DBG(cat,img);
						
							temp_ind = Y_IND(cat,img);
							// supervised term:
							//... sigma approximations ...
							grad -= temp_F_prod_all * Y[temp_ind];
		
							// unsupervised term:
							grad +=  temp_F_prod_all * pred[temp_ind];
						}
					}
						
		}}}}}}}
		F3_IND_DBG(f3_, f2_, a3_x_, a3_y_)
		F3[F3_IND(f3_, f2_, a3_x_, a3_y_)] -= eps_F3*2*grad;
	}}
	printf("F3 grad: %i sec\n", (unsigned)time(NULL) - t_start);
	
	printf("%f, cost %f, F1 grad: %i sec\n", grad, compute_cost(), (unsigned)time(NULL) - t_start);
	} // gradient steps
	return 0;
}


