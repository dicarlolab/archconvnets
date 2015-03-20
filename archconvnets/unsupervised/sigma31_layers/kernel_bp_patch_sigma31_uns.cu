#define FL_IND_C1(A,B,C,D)(D + (N_C)*C + (N_C*max_output_sz3)*B + (N_C*max_output_sz3*max_output_sz3)*A)

__global__ void kernel_bp_patch_sigma31_uns(float * deriv, float * sigma_imgs, float * imgs, float * F1, float * F2, float * F3, float * FL, 
	long * output_switches3_x_s31, long * output_switches3_y_s31, long * output_switches2_x_s31, long * output_switches2_y_s31, long * output_switches1_x_s31, long * output_switches1_y_s31,
	long * output_switches3_x, long * output_switches3_y, long * output_switches2_x, long * output_switches2_y, long * output_switches1_x, long * output_switches1_y,
	int N_IMGS, int N_C, int n0, int n1, int n2, int n3, int s1, int s2, int s3, int max_output_sz3, IND_DTYPE max_output_sz3_max_output_sz3,	
	IND_DTYPE max_output_sz3_max_output_sz3_n3, IND_DTYPE max_output_sz2_max_output_sz2, IND_DTYPE max_output_sz2_max_output_sz2_n2, IND_DTYPE max_output_sz1_max_output_sz1,
	IND_DTYPE max_output_sz1_max_output_sz1_n1, IND_DTYPE img_sz_img_sz_3, IND_DTYPE img_sz_img_sz, int deriv_ind, IND_DTYPE max_output_sz2, IND_DTYPE max_output_sz1, float * pred, int img_sz){
	
	int r = blockIdx.x;
		
	int f1, channel, a1_x, a1_y, f2, a2_x, a2_y, f3, a3_x, a3_y, z1, z2, cat, img;
	int a3_x_global, a3_y_global, a2_x_global, a2_y_global, a1_x_global, a1_y_global;
	
	float F_prod;
	IND_DTYPE deriv_in_ind;
	
	int s1_s1 = s1*s1;
	
	int cat_sz = N_C;
	int f1_sz = n1;
	int f2_sz = n2;
	int f3_sz = n3;
	int channel_sz = 3;
	int a1_x_sz = s1;
	int a1_y_sz = s1;
	int a2_x_sz = s2;
	int a2_y_sz = s2;
	int a3_x_sz = s3;
	int a3_y_sz = s3;
	int z1_sz = max_output_sz3;
	int z2_sz = max_output_sz3;
	int img_sz_ind = N_IMGS;
	
	int * cat_i = &cat;
	int * f1_i = &f1;
	int * f2_i = &f2;
	int * f3_i = &f3;
	int * channel_i = &channel;
	int * a1_x_i = &a1_x;
	int * a1_y_i = &a1_y;
	int * a2_x_i = &a2_x;
	int * a2_y_i = &a2_y;
	int * a3_x_i = &a3_x;
	int * a3_y_i = &a3_y;
	int * z1_i = &z1;
	int * z2_i = &z2;
	int * img_i = &img;
	
	
	float temp_deriv = 0;
	float F_prod_pred;
	float F32_imgs, F321;
	
	int switches_3_ind;
	int switches_2_ind;
	int switches_1_ind;
	
	int o3_ind;
	int o2_ind;
	int o1_ind;
	
	float FL32_pred_t;
	float F3_t;
	int f1_ind;
	int fL_ind;
	int i_ind;
	
	int output_switches2_xt;
	int output_switches2_yt;
	
	int output_switches1_xt;
	int output_switches1_yt;
	
	float imgs_t;
	
	deriv_in_ind = r;
	/////////// which loops to unravel across the grid
	if(deriv_ind == 1){
		int f1_c = r / (3*s1*s1);
		r = r % (3*s1*s1);
		f1_i = &f1_c;
		f1_sz = 1;
		
		int channel_c = r / (s1*s1);
		r = r % (s1*s1);
		channel_i = &channel_c;
		channel_sz = 1;
		
		int a1_x_c = r / s1;
		int a1_y_c = r % s1;
		a1_x_i = &a1_x_c;
		a1_y_i = &a1_y_c;
		a1_x_sz = 1;
		a1_y_sz = 1;
		
		int img_c = threadIdx.x;
		img_i = &img_c;
		img_sz_ind = 1;
		
		int a2_y_c = threadIdx.y;
		a2_y_i = &a2_y_c;
		a2_y_sz = 1;
		
		int f3_c = blockIdx.y;
		f3_i = &f3_c;
		f3_sz = 1;
		
		int f2_c = blockIdx.z;
		f2_i = &f2_c;
		f2_sz = 1;
		
		/*r = blockIdx.z;
		int z1_c = r / max_output_sz3;
		z1_i = &z1_c;
		z1_sz = 1;
		
		int z2_c = r % max_output_sz3;
		z2_i = &z2_c;
		z2_sz = 1;*/
	
		int pred_ind = N_C * *img_i;
		
		for(f3=0; f3 < f3_sz; f3++){ for(z1=0; z1 < z1_sz; z1++){ for(z2=0; z2 < z2_sz; z2++){ 
			
			fL_ind = FL_IND_C1(*f3_i,*z1_i,*z2_i,0);
			
			for(a3_x=0; a3_x < a3_x_sz; a3_x++){ for(a3_y=0; a3_y < a3_y_sz; a3_y++){
			
			///////////////////////////////////////////////////// uns
			// pool3 -> conv3
			o3_ind = O3_IND(*img_i,*f3_i,*z1_i,*z2_i);
			a3_x_global = output_switches3_x[o3_ind] + *a3_x_i;
			a3_y_global = output_switches3_y[o3_ind] + *a3_y_i;
			
			for(f2=0; f2 < f2_sz; f2++){ 
			
				F3_t = F3[F3_IND(*f3_i, *f2_i, *a3_x_i, *a3_y_i)];
				
				o2_ind = O2_IND(*img_i,*f2_i,a3_x_global,a3_y_global);
				output_switches2_xt = output_switches2_x[o2_ind];
				output_switches2_yt = output_switches2_y[o2_ind];
				
				a2_y_global = output_switches2_yt + *a2_y_i;
				
				for(a2_x=0; a2_x < a2_x_sz; a2_x++){
				
					o1_ind = O1_IND(*img_i,*f1_i, output_switches2_xt + *a2_x_i, a2_y_global);

					F32_imgs = F2[F2_IND(*f2_i, *f1_i, *a2_x_i, *a2_y_i)] * F3_t * imgs[I_IND(*img_i, *channel_i,     output_switches1_x[o1_ind] + *a1_x_i,    output_switches1_y[o1_ind] + *a1_y_i)];
						
					for(cat=0; cat < cat_sz; cat++){ 
						temp_deriv -= FL[*cat_i + fL_ind] * F32_imgs * pred[*cat_i + pred_ind];

		}}}}}}}}
	}
	
	atomicAdd(&deriv[deriv_in_ind], temp_deriv);
	return;
}