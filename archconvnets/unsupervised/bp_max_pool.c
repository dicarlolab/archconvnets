// max pool inputs from conv_output[], store in max_output[] and switch_output[]
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
		C_IND_DBG(filter, x + x_loc, y + y_loc, img)
		
		temp_compare_ind = C_IND(filter, x + x_loc, y + y_loc, img);
		temp_compare = conv_output[temp_compare_ind];
		if(temp_max < temp_compare){
			temp_max = temp_compare;
			temp_max_ind = temp_compare_ind;
		}
	     } // y_loc
	    } // x_loc
	    O_IND_DBG(filter, x_ind, y_ind, img)

	    max_output[O_IND(filter, x_ind, y_ind, img)] = temp_max;
	    switch_output[O_IND(filter, x_ind, y_ind, img)] = temp_max_ind;
	    
	    y_ind ++;
	   } // y
	   x_ind ++;
	  } // x
	 } // filter
	} // img
}
