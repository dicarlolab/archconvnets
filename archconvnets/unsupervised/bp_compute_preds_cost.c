// given max_output3 and FL, compute label predictions for each image, store in pred[]
inline void compute_preds(){
	memset(pred, 0, N_C * N_IMGS * sizeof(float));
	for(int cat = 0; cat < N_C; cat++){
		for(int img = 0; img < N_IMGS; img++){
			for(int filter = 0; filter < n3; filter++){
				for(int x = 0; x < max_output_sz3; x++){
					for(int y = 0; y < max_output_sz3; y++){
						Y_IND_DBG(cat, img)
						FL_IND_DBG(cat, filter, x, y)
						SW3_IND_DBG(filter, x, y, img)
						
						pred[Y_IND(cat, img)] += FL[FL_IND(cat, filter, x, y)] * 
							max_output3[SW3_IND(filter, x, y, img)];
					}
				}
			}
		}
	}
}

//given pred and y, compute sum squarred error
inline float compute_cost(){
	float cost = 0;
	float temp;
	for(int cat = 0; cat < N_C; cat++){
		for(int img = 0; img < N_IMGS; img++){
			Y_IND_DBG(cat,img)
		
			temp = (pred[Y_IND(cat,img)] - Y[Y_IND(cat,img)]);
			cost += temp*temp;
		}
	}
	return cost;
}
