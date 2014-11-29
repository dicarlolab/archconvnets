
// return pixel based on max switch locations (i.e., unpooling down to a single pixel)
// this function is called many, many times in gradient computation so needs to be very fast...
float return_px(int f1, int f2, int f3, int z1, int z2, int a3_x, int a3_y, int a2_x, int a2_y, int a1_x, int a1_y, int channel, int img){
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
	
	SW3_IND_DBG(f3, z1, z2, img)
	
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
	#ifdef DEBUG
	if((r - a3_x_global*n3) != f3) PANIC("indexing problem in return_px(). switch_output3 filter index is incorrect")
	#endif
	//printf("%i, %i, %i, %i, %i\n", (r - a3_x_global*n3), a3_x_global, a3_y_global, img, ind);
	//printf("(%i, %i, %i)\n", n3, output_sz3, N_IMGS);
	
	// SW2_IND(...) = pool2 index
	SW2_IND_DBG(f2, a3_x_global + a3_x, a3_y_global + a3_y, img)
	ind = switch_output2[SW2_IND(f2, a3_x_global + a3_x, a3_y_global + a3_y, img)]; // pool2 -> conv2 index

	#ifdef DEBUG
	/*int t = (ind / (n2*output_sz2*output_sz2));
	if(t != img)
                //printf("ind %i r %i img %i output_sz2 %i n2 %i\n", ind, (ind / (n2*output_sz2*output_sz2)), img, output_sz2, n2);
		printf("test %i %i\n", t, img);
		 printf("test %i %i\n", t, img);
 printf("test %i %i\n", t, img);
		printf("ind %i r %i img %i output_sz2 %i n2 %i\n", ind, (ind / (n2*output_sz2*output_sz2)), img, output_sz2, n2);
//		PANIC("indexing problem in return_px(). switch_output2 img index is incorrect")*/
	#endif
	
	r = ind - img*n2*output_sz2*output_sz2;
	a2_y_global = r / (n2*output_sz2);
	r -= a2_y_global*(n2*output_sz2);
	a2_x_global = r / n2;
	#ifdef DEBUG
	if((r - a2_x_global*n2) != f2) PANIC("indexing problem in return_px(). switch_output2 filter index is incorrect")
	#endif
		
	// SW1_IND(...) = pool1 index
	SW1_IND_DBG(f1, a2_x_global + a2_x, a2_y_global + a2_y, img)
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
	
	IMG_IND_DBG(channel, a1_x_global*STRIDE1 + a1_x, a1_y_global*STRIDE1 + a1_y, img)
	return imgs[IMG_IND(channel, a1_x_global*STRIDE1 + a1_x, a1_y_global*STRIDE1 + a1_y, img)];
}
