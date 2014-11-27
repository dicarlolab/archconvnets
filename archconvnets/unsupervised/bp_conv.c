// convolve filters[] with imgs[], store outputs in conv_output[]
// output_sz: size of convolutional output in px
void conv(float *filters, float *conv_output, int n_filters, int n_channels, int filter_sz, int img_sz, float *imgs, int stride, int output_sz){
	memset(conv_output, 0, n_filters * output_sz * output_sz * N_IMGS * sizeof(float));
	for(int img = 0; img < N_IMGS; img ++){
	 for(int filter = 0; filter < n_filters; filter++){
       for(int channel = 0; channel < n_channels; channel ++){
	   for(int x = 0; x < output_sz; x ++){
	    for(int y = 0; y < output_sz; y ++){
	     for(int x_loc = 0; x_loc < filter_sz; x_loc ++){
	      for(int y_loc = 0; y_loc < filter_sz; y_loc ++){
			O_IND_DBG(filter, x, y, img)
			F_IND_DBG(filter, channel, x_loc, y_loc)
			I_IND_DBG(channel, stride*x + x_loc, stride*y + y_loc, img)
		
			conv_output[O_IND(filter, x, y, img)] += 
					filters[F_IND(filter, channel, x_loc, y_loc)] * imgs[I_IND(channel, stride*x + x_loc, stride*y + y_loc, img)];
	      } // y_loc
	     } // x_loc
	    } // y
	   } // x
	  } // channel
	 } // filter
	} // img
	return;
}
