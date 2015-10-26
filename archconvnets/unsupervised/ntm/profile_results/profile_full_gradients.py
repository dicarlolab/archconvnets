Timer unit: 1e-06 s

Total time: 43.1552 s
File: ntm_core.py
Function: full_gradients at line 306

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   306                                           
   307                                           
   308                                           ### compute full gradients from state partials
   309       615       117050    190.3      0.3  # 24.8 of main()
   310                                           def full_gradients(read_mem, t, mem_prev, DOR_DWR, DOR_DBR, DOR_DWW, DOR_DBW, DOR_DWUNDER,DOR_DBUNDER, OR, DMEM_PREV_DWW, \
   311                                           			DMEM_PREV_DBW, DMEM_PREV_DWUNDER, DMEM_PREV_DBUNDER):
   312       615        21817     35.5      0.1  	derr_dread_mem = sq_points_dinput(read_mem - t)
   313       615       513934    835.7      1.2  	
   314                                           	# read weights
   315       615      7886632  12823.8     18.3  	dread_mem_dor = linear_F_dF_nsum(mem_prev)
   316       615      1089110   1770.9      2.5  	derr_dor = mult_partials(derr_dread_mem, dread_mem_dor, read_mem)
   317       615     16621584  27027.0     38.5  	
   318       615      2123334   3452.6      4.9  	DWR = mult_partials_collapse__layers(derr_dor, DOR_DWR, OR[F]) # 18.3%
   319       615      2701915   4393.4      6.3  	DBR = mult_partials_collapse__layers(derr_dor, DOR_DBR, OR[F])
   320       615       373229    606.9      0.9  	DWW = mult_partials_collapse__layers(derr_dor, DOR_DWW, OR[F]) # 38.5%
   321                                           	DBW = mult_partials_collapse__layers(derr_dor, DOR_DBW, OR[F])
   322                                           	DWUNDER = mult_partials_collapse__layers(derr_dor, DOR_DWUNDER, OR[F])
   323       615        19892     32.3      0.0  	DBUNDER = mult_partials_collapse__layers(derr_dor, DOR_DBUNDER, OR[F])
   324       615       293210    476.8      0.7  	
   325                                           	# write weights
   326       615      8634156  14039.3     20.0  	dread_mem_dmem_prev = linear_F_dx_nsum(OR[F])
   327       615      1142135   1857.1      2.6  	derr_dmem_prev = mult_partials(derr_dread_mem, dread_mem_dmem_prev, read_mem)
   328       615      1394175   2267.0      3.2  	
   329       615       222153    361.2      0.5  	DWW = mult_partials_collapse__layers(derr_dmem_prev, DMEM_PREV_DWW, mem_prev, DWW) # 20%
   330                                           	DBW = mult_partials_collapse__layers(derr_dmem_prev, DMEM_PREV_DBW, mem_prev, DBW)
   331       615          851      1.4      0.0  	DWUNDER = mult_partials_collapse__layers(derr_dmem_prev, DMEM_PREV_DWUNDER, mem_prev, DWUNDER)
   332                                           	DBUNDER = mult_partials_collapse__layers(derr_dmem_prev, DMEM_PREV_DBUNDER, mem_prev, DBUNDER)
   333                                           	
   334                                           	return DWR, DBR, DWW, DBW, DWUNDER, DBUNDER

