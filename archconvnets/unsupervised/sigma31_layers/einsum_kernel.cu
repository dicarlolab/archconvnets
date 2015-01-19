#define S31_IND2(A,B,C,D,E,F,G,H,I,J,K,L,M)((M)*z2b + (L)*max_output_sz3s + (K)*max_output_sz3_max_output_sz3s + (J)*max_output_sz3_max_output_sz3_s3s + (I)*max_output_sz3_max_output_sz3_s3_s3s + \
	(H)*max_output_sz3_max_output_sz3_s3_s3_n3s + (G)*max_output_sz3_max_output_sz3_s3_s3_n3_s2s + (F)*max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2s + (E)*max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2s + \
	(D)*max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1s + (C)*max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1s + (B)*max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1_n0s + \
	(A)*max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1_n0_n1s)
#define F1_IND(A,B,C,D)(D + (s1)*C + (s1*s1)*B + (s1*s1*n0)*A)
#define F2_IND(A,B,C,D)(D + (s2)*C + (s2*s2)*B + (s2*s2*n1)*A)
#define F3_IND(A,B,C,D)(D + (s3)*C + (s3*s3)*B + (s3*s3*n2)*A)
#define FL_IND(A,B,C,D)(D + (max_output_sz3)*C + (max_output_sz3*max_output_sz3)*B + (max_output_sz3*max_output_sz3*n3)*A)
	
__global__ void kernel_deriv(float * sum_res, float * sigma31, float * F1, float * F2, float * F3, float * FL,
		IND_DTYPE max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1_n0_n1s, IND_DTYPE max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1_n0s,
		IND_DTYPE max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1s, IND_DTYPE max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1s, IND_DTYPE max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2s,
		IND_DTYPE max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2s, IND_DTYPE max_output_sz3_max_output_sz3_s3_s3_n3_s2s, IND_DTYPE max_output_sz3_max_output_sz3_s3_s3_n3s, IND_DTYPE max_output_sz3_max_output_sz3_s3_s3s,
		IND_DTYPE max_output_sz3_max_output_sz3_s3s, IND_DTYPE max_output_sz3_max_output_sz3s, int z2b, int n0, int n0s, int n1, int n1s, int n2, int n2s, int n3, int n3s,
		int max_output_sz3, int max_output_sz3s, int s1, int s1s, int s2, int s2s, int s3, int s3s, int N_C, int deriv_ind){
	
	extern __shared__ float sum_res_shared[];
	if(threadIdx.x == 0){
		*sum_res_shared = 0;
	}
	__syncthreads();
	
	int f1, f0;
	int s1x, s1y;
	int f2;
	int s2x, s2y;
	int f3;
	int s3x, s3y;
	int z1, z2;
	int cat_i, cat_j;
	
	////////////////////////////////////////////////////////////////////////////////////////////////////
	// which dimensions have been unraveled across the *grid* and that we should not loop over here? (we are solving for the term containing these particular indices)
	int *f1i = &f1, *f0i = &f0;
	int *s1xi = &s1x, *s1yi = &s1y;
	int *f2i = &f2;
	int *s2xi = &s2x, *s2yi = &s2y;
	int *f3i = &f3;
	int *s3xi = &s3x, *s3yi = &s3y;
	int *z1i = &z1, *z2i = &z2;
	int *cat_ii = &cat_i, *cat_ji = &cat_j;
	
	int f1_sz = n1;
	int f0_sz = n0;
	int s1x_sz = s1;
	int s1y_sz = s1;
	int f2_sz = n2;
	int s2x_sz = s2;
	int s2y_sz = s2;
	int f3_sz = n3;
	int s3x_sz = s3;
	int s3y_sz = s3;
	int z1_sz = max_output_sz3;
	int z2_sz = max_output_sz3;
	int cat_i_sz = N_C;
	int cat_j_sz = N_C;
	int output_ind;
	
	int r = blockIdx.x;
	int t = threadIdx.x;
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// which indices are raveled across the grid and threads?
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// prediction
	if(deriv_ind == 0){
		///////////////////////////////////////////// indices that we keep (specify output term)
		int cat_jc = r;
		cat_ji = &cat_jc;
		cat_j_sz = 1;
		
		int cat_ic = blockIdx.y;
		cat_ii = &cat_ic;
		cat_i_sz = 1;
	
		output_ind = cat_jc*N_C + cat_ic;
		
		//////////////////////////////////////// indices that are raveled over the threads
		//int f0c = threadIdx.y;
		//f0i = &f0c;
		//f0_sz = 1;
		
		int s1xc = t / (s2*s2*s3);
		s1xi = &s1xc;
		t = t % (s2*s2*s3);
		s1x_sz = 1;
		
		int s2xc = t / (s2*s3);
		s2xi = &s2xc;
		t = t % (s2*s3);
		s2x_sz = 1;
		
		int s2yc = t / s3;
		s2yi = &s2yc;
		s2y_sz = 1;
		
		int s3xc = t % s3;
		s3xi = &s3xc;
		s3x_sz = 1;
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// F1 deriv
	}else if(deriv_ind == 1){
		///////////////////////////////////////////// indices that we keep (specify output term)
		int cat_jc = r / (N_C*s1*s1);
		cat_ji = &cat_jc;
		r = r % (N_C*s1*s1);
		cat_j_sz = 1;
		
		int cat_ic = r / (s1*s1);
		cat_ii = &cat_ic;
		r = r % (s1*s1);
		cat_i_sz = 1;
	
		int s1xc = r / s1;
		s1xi = &s1xc;
		s1x_sz = 1;
		
		int s1yc = r % s1;
		s1yi = &s1yc;
		s1y_sz = 1;
		
		/////////////////////////////
		int f1c = blockIdx.y;
		f1i = &f1c;
		f1_sz = 1;
		
		int f0c = blockIdx.z;
		f0i = &f0c;
		f0_sz = 1;
		
		output_ind = cat_jc*(N_C*n1*n0*s1*s1) + cat_ic*(n1*n0*s1*s1) + f1c*(n0*s1*s1) + f0c*(s1*s1) + s1xc*s1 + s1yc;
		
		//////////////////////////////////////// indices that are raveled over the threads
		int s2xc = t / (s2*s3*s3);
		s2xi = &s2xc;
		t = t % (s2*s3*s3);
		s2x_sz = 1;
		
		int s2yc = t / (s3*s3);
		s2yi = &s2yc;
		t = t % (s3*s3);
		s2y_sz = 1;
		
		int s3xc = t / s3;
		s3xi = &s3xc;
		s3x_sz = 1;
		
		int s3yc = t % s3;
		s3yi = &s3yc;
		s3y_sz = 1;
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// F2 deriv
	}else if(deriv_ind == 2){
		///////////////////////////////////////////// indices that we keep (specify output term)
		int cat_jc = r / (N_C*s2*s2);
		cat_ji = &cat_jc;
		r = r % (N_C*s2*s2);
		cat_j_sz = 1;
		
		int cat_ic = r / (s2*s2);
		cat_ii = &cat_ic;
		r = r % (s2*s2);
		cat_i_sz = 1;
	
		int s2xc = r / s2;
		s2xi = &s2xc;
		s2x_sz = 1;
		
		int s2yc = r % s2;
		s2yi = &s2yc;
		s2y_sz = 1;
		
		///////////////////////////////////////
		int f2c = blockIdx.y;
		f2i = &f2c;
		f2_sz = 1;
		
		int f1c = blockIdx.z;
		f1i = &f1c;
		f1_sz = 1;
		
		output_ind = cat_jc*(N_C*n2*n1*s2*s2) + cat_ic*(n2*n1*s2*s2) + f2c*(n1*s2*s2) + f1c*(s2*s2) + s2xc*s2 + s2yc;
		
		//////////////////////////////////////// indices that are raveled over the threads
		int s1xc = t / (s1*s3*s3);
		s1xi = &s1xc;
		t = t % (s1*s3*s3);
		s1x_sz = 1;
		
		int s1yc = t / (s3*s3);
		s1yi = &s1yc;
		t = t % (s3*s3);
		s1y_sz = 1;
		
		int s3xc = t / s3;
		s3xi = &s3xc;
		s3x_sz = 1;
		
		int s3yc = t % s3;
		s3yi = &s3yc;
		s3y_sz = 1;
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// F3 deriv
	}else if(deriv_ind == 3){
		///////////////////////////////////////////// indices that we keep (specify output term)
		int cat_jc = r / (N_C*s3*s3);
		cat_ji = &cat_jc;
		r = r % (N_C*s3*s3);
		cat_j_sz = 1;
		
		int cat_ic = r / (s3*s3);
		cat_ii = &cat_ic;
		r = r % (s3*s3);
		cat_i_sz = 1;
	
		int s3xc = r / s3;
		s3xi = &s3xc;
		s3x_sz = 1;
		
		int s3yc = r % s3;
		s3yi = &s3yc;
		s3y_sz = 1;
		
		///////////////////////////////////////
		int f3c = blockIdx.y;
		f3i = &f3c;
		f3_sz = 1;
		
		int f2c = blockIdx.z;
		f2i = &f2c;
		f2_sz = 1;
		
		output_ind = cat_jc*(N_C*n3*n2*s3*s3) + cat_ic*(n3*n2*s3*s3) + f3c*(n2*s3*s3) + f2c*(s3*s3) + s3xc*s3 + s3yc;
		
		//////////////////////////////////////// indices that are raveled over the threads
		int s1xc = t / (s1*s2);
		s1xi = &s1xc;
		t = t % (s1*s2);
		s1x_sz = 1;
		
		int s1yc = t / (s2);
		s1yi = &s1yc;
		t = t % (s2);
		s1y_sz = 1;
		
		int s2xc = t ;/// s2;
		s2xi = &s2xc;
		s2x_sz = 1;
		
		/*int s2yc = t % s2;
		s2yi = &s2yc;
		s2y_sz = 1;*/
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// FL deriv
	}else if(deriv_ind == 4){
		///////////////////////////////////////////// indices that we keep (specify output term)
		int cat_ic = r / max_output_sz3;
		cat_ii = &cat_ic;
		cat_i_sz = 1;
		
		int z1c = r % max_output_sz3;
		z1i = &z1c;
		z1_sz = 1;
		
		///////////////////////////////////////
		int z2c = blockIdx.y;
		z2i = &z2c;
		z2_sz = 1;
		
		int f3c = blockIdx.z;
		f3i = &f3c;
		f3_sz = 1;
		
		output_ind = cat_ic*(n3*max_output_sz3*max_output_sz3) + f3c*(max_output_sz3*max_output_sz3) + z1c*max_output_sz3 + z2c;
		
		
		// we want cat_j = cat_i, because we do not need to compute the products sigma31[cat_j] * FL[cat_i] for all cat_i,cat_j (because FL is 1)
		cat_ji = &cat_ic;
		cat_j_sz = 1;
		
		//////////////////////////////////////// indices that are raveled over the threads
		
		
		int s1xc = t / (s1*s2);
		s1xi = &s1xc;
		t = t % (s1*s2);
		s1x_sz = 1;
		
		int s1yc = t / (s2);
		s1yi = &s1yc;
		t = t % (s2);
		s1y_sz = 1;
		
		int s2xc = t ;/// s2;
		s2xi = &s2xc;
		s2x_sz = 1;
		
		/*int s2yc = t % s2;
		s2yi = &s2yc;
		s2y_sz = 1;*/
	}
	
	float sum_res_local = 0;
	//////////////////////////////////////////////////////////////////////////////// do the right summation (exclude the relevant filter values from the product or not)
	if(deriv_ind == 0){
		for(cat_i = 0; cat_i < cat_i_sz; cat_i++){
			for(cat_j = 0; cat_j < cat_j_sz; cat_j++){
				for(f1 = 0; f1 < f1_sz; f1++){
					for(f0 = 0; f0 < f0_sz; f0++){
						for(s1x = 0; s1x < s1x_sz; s1x++){
							for(s1y = 0; s1y < s1y_sz; s1y++){
								for(f2 = 0; f2 < f2_sz; f2++){
									for(s2x = 0; s2x < s2x_sz; s2x++){
										for(s2y = 0; s2y < s2y_sz; s2y++){
											for(f3 = 0; f3 < f3_sz; f3++){
												for(s3x = 0; s3x < s3x_sz; s3x++){
													for(s3y = 0; s3y < s3y_sz; s3y++){
														for(z1 = 0; z1 < z1_sz; z1++){ 
															for(z2 = 0; z2 < z2_sz; z2++){
																sum_res_local += sigma31[S31_IND2(*cat_ii, *f1i, *f0i, *s1xi, *s1yi, *f2i, *s2xi, *s2yi, *f3i, *s3xi, *s3yi, *z1i, *z2i)] *
																	F1[F1_IND(*f1i, *f0i, *s1xi, *s1yi)] * F2[F2_IND(*f2i, *f1i, *s2xi, *s2yi)] * F3[F3_IND(*f3i, *f2i, *s3xi, *s3yi)] * FL[FL_IND(*cat_ji, *f3i, *z1i, *z2i)];
															} // z2
														} // z1
													}
												} // s3x, s3y
											} // f3
										}
									} // s2x, s2y
								} // f2
							}
						} // s1x, s1y
					} // f0
				} // f1
			} // cat_j
		} // cat_i
	}else if(deriv_ind == 1){ ////////////////////////////////////////////////////////////////////////////////// exclude F1:
		for(cat_i = 0; cat_i < cat_i_sz; cat_i++){
			for(cat_j = 0; cat_j < cat_j_sz; cat_j++){
				for(f1 = 0; f1 < f1_sz; f1++){
					for(f0 = 0; f0 < f0_sz; f0++){
						for(s1x = 0; s1x < s1x_sz; s1x++){
							for(s1y = 0; s1y < s1y_sz; s1y++){
								for(f2 = 0; f2 < f2_sz; f2++){
									for(s2x = 0; s2x < s2x_sz; s2x++){
										for(s2y = 0; s2y < s2y_sz; s2y++){
											for(f3 = 0; f3 < f3_sz; f3++){
												for(s3x = 0; s3x < s3x_sz; s3x++){
													for(s3y = 0; s3y < s3y_sz; s3y++){
														for(z1 = 0; z1 < z1_sz; z1++){ 
															for(z2 = 0; z2 < z2_sz; z2++){
																sum_res_local += sigma31[S31_IND2(*cat_ii, *f1i, *f0i, *s1xi, *s1yi, *f2i, *s2xi, *s2yi, *f3i, *s3xi, *s3yi, *z1i, *z2i)] *
																	F2[F2_IND(*f2i, *f1i, *s2xi, *s2yi)] * F3[F3_IND(*f3i, *f2i, *s3xi, *s3yi)] * FL[FL_IND(*cat_ji, *f3i, *z1i, *z2i)];
															} // z2
														} // z1
													}
												} // s3x, s3y
											} // f3
										}
									} // s2x, s2y
								} // f2
							}
						} // s1x, s1y
					} // f0
				} // f1
			} // cat_j
		} // cat_i
	}else if(deriv_ind == 2){ ////////////////////////////////////////////////////////////////////////////////// exclude F2:
		for(cat_i = 0; cat_i < cat_i_sz; cat_i++){
			for(cat_j = 0; cat_j < cat_j_sz; cat_j++){
				for(f1 = 0; f1 < f1_sz; f1++){
					for(f0 = 0; f0 < f0_sz; f0++){
						for(s1x = 0; s1x < s1x_sz; s1x++){
							for(s1y = 0; s1y < s1y_sz; s1y++){
								for(f2 = 0; f2 < f2_sz; f2++){
									for(s2x = 0; s2x < s2x_sz; s2x++){
										for(s2y = 0; s2y < s2y_sz; s2y++){
											for(f3 = 0; f3 < f3_sz; f3++){
												for(s3x = 0; s3x < s3x_sz; s3x++){
													for(s3y = 0; s3y < s3y_sz; s3y++){
														for(z1 = 0; z1 < z1_sz; z1++){ 
															for(z2 = 0; z2 < z2_sz; z2++){
																sum_res_local += sigma31[S31_IND2(*cat_ii, *f1i, *f0i, *s1xi, *s1yi, *f2i, *s2xi, *s2yi, *f3i, *s3xi, *s3yi, *z1i, *z2i)] *
																	F1[F1_IND(*f1i, *f0i, *s1xi, *s1yi)] * F3[F3_IND(*f3i, *f2i, *s3xi, *s3yi)] * FL[FL_IND(*cat_ji, *f3i, *z1i, *z2i)];
															} // z2
														} // z1
													}
												} // s3x, s3y
											} // f3
										}
									} // s2x, s2y
								} // f2
							}
						} // s1x, s1y
					} // f0
				} // f1
			} // cat_j
		} // cat_i
	}else if(deriv_ind == 3){ ////////////////////////////////////////////////////////////////////////////////// exclude F3:
		for(cat_i = 0; cat_i < cat_i_sz; cat_i++){
			for(cat_j = 0; cat_j < cat_j_sz; cat_j++){
				for(f1 = 0; f1 < f1_sz; f1++){
					for(f0 = 0; f0 < f0_sz; f0++){
						for(s1x = 0; s1x < s1x_sz; s1x++){
							for(s1y = 0; s1y < s1y_sz; s1y++){
								for(f2 = 0; f2 < f2_sz; f2++){
									for(s2x = 0; s2x < s2x_sz; s2x++){
										for(s2y = 0; s2y < s2y_sz; s2y++){
											for(f3 = 0; f3 < f3_sz; f3++){
												for(s3x = 0; s3x < s3x_sz; s3x++){
													for(s3y = 0; s3y < s3y_sz; s3y++){
														for(z1 = 0; z1 < z1_sz; z1++){ 
															for(z2 = 0; z2 < z2_sz; z2++){
																sum_res_local += sigma31[S31_IND2(*cat_ii, *f1i, *f0i, *s1xi, *s1yi, *f2i, *s2xi, *s2yi, *f3i, *s3xi, *s3yi, *z1i, *z2i)] *
																	F1[F1_IND(*f1i, *f0i, *s1xi, *s1yi)] * F2[F2_IND(*f2i, *f1i, *s2xi, *s2yi)] * FL[FL_IND(*cat_ji, *f3i, *z1i, *z2i)];
															} // z2
														} // z1
													}
												} // s3x, s3y
											} // f3
										}
									} // s2x, s2y
								} // f2
							}
						} // s1x, s1y
					} // f0
				} // f1
			} // cat_j
		} // cat_i
	}else if(deriv_ind == 4){ ////////////////////////////////////////////////////////////////////////////////// exclude FL:
		for(cat_i = 0; cat_i < cat_i_sz; cat_i++){
			for(cat_j = 0; cat_j < cat_j_sz; cat_j++){
				for(f1 = 0; f1 < f1_sz; f1++){
					for(f0 = 0; f0 < f0_sz; f0++){
						for(s1x = 0; s1x < s1x_sz; s1x++){
							for(s1y = 0; s1y < s1y_sz; s1y++){
								for(f2 = 0; f2 < f2_sz; f2++){
									for(s2x = 0; s2x < s2x_sz; s2x++){
										for(s2y = 0; s2y < s2y_sz; s2y++){
											for(f3 = 0; f3 < f3_sz; f3++){
												for(s3x = 0; s3x < s3x_sz; s3x++){
													for(s3y = 0; s3y < s3y_sz; s3y++){
														for(z1 = 0; z1 < z1_sz; z1++){ 
															for(z2 = 0; z2 < z2_sz; z2++){
																sum_res_local += sigma31[S31_IND2(*cat_ii, *f1i, *f0i, *s1xi, *s1yi, *f2i, *s2xi, *s2yi, *f3i, *s3xi, *s3yi, *z1i, *z2i)] *
																	F1[F1_IND(*f1i, *f0i, *s1xi, *s1yi)] * F2[F2_IND(*f2i, *f1i, *s2xi, *s2yi)] * F3[F3_IND(*f3i, *f2i, *s3xi, *s3yi)];
															} // z2
														} // z1
													}
												} // s3x, s3y
											} // f3
										}
									} // s2x, s2y
								} // f2
							}
						} // s1x, s1y
					} // f0
				} // f1
			} // cat_j
		} // cat_i
	}

	atomicAdd(&sum_res_shared[0], sum_res_local);
	
	__syncthreads();
	if(threadIdx.x == 0)
		sum_res[output_ind] = *sum_res_shared;
	//atomicAdd(&sum_res[blockIdx.x], sum_res_local);
}
