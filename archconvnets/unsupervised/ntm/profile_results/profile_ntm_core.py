Timer unit: 1e-06 s

Total time: 174.023 s
File: ntm_core.py
Function: reverse_pass_partials at line 256

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   256                                           @profile
   257                                           def reverse_pass_partials(WUNDER,BUNDER, WR,WW,BR,BW, OUNDER, OUNDER_PREV, OR, OR_PREV, OW_PREV, \
   258                                           				OW_PREV_PREV, mem_prev, mem_prev_prev, x, x_prev, frame, DOW_DWW, DOW_DBW, \
   259                                           				DOW_DWUNDER, DOW_DBUNDER, DMEM_PREV_DWW, DMEM_PREV_DBW, DMEM_PREV_DWUNDER, \
   260                                           				DMEM_PREV_DBUNDER, DOR_DWR, DOR_DBR, DOR_DWUNDER, DOR_DBUNDER, DOR_DWW, DOR_DBW):
   261       820       511823    624.2      0.3  	dor_dgsharpen = dsharpen_dw(OR[SHIFTED], OR[GAMMA])
   262       820       503512    614.0      0.3  	dow_prev_dgsharpen = dsharpen_dw(OW_PREV[SHIFTED], OW_PREV[GAMMA])
   263                                           	
   264       820       121791    148.5      0.1  	dgsharpen_dor_in = shift_w_dw_interp_nsum(OR[SHIFT])
   265       820       120455    146.9      0.1  	dgsharpen_dow_prev_in = shift_w_dw_interp_nsum(OW_PREV[SHIFT])
   266                                           	
   267       820       398089    485.5      0.2  	dor_dor_in = mult_partials(dor_dgsharpen, dgsharpen_dor_in, OR[SHARPENED])
   268       820       396515    483.6      0.2  	dow_prev_dow_prev_in = mult_partials(dow_prev_dgsharpen, dgsharpen_dow_prev_in, OW_PREV[SHARPENED])
   269                                           	
   270                                           	# partials for write head output (OW)
   271       820         1208      1.5      0.0  	if frame > 1:
   272       818         1014      1.2      0.0  		DOW_DWW, DOW_DBW, DOW_DWUNDER, DOW_DBUNDER = do_dw__o_prev(WW, OW_PREV_PREV[F], DOW_DWW, DOW_DBW, DOW_DWUNDER,\
   273       818     23307595  28493.4     13.4  											DOW_DBUNDER, OW_PREV, dow_prev_dow_prev_in)
   274       818         2294      2.8      0.0  		DOW_DWW, DOW_DBW, DOW_DWUNDER, DOW_DBUNDER = do_dw__mem_prev(WW, DOW_DWW, DOW_DBW, DOW_DWUNDER, DOW_DBUNDER, OW_PREV, \
   275       818          741      0.9      0.0  											mem_prev_prev, DMEM_PREV_DWW, DMEM_PREV_DBW, DMEM_PREV_DWUNDER, \
   276       818     17981155  21981.9     10.3  											DMEM_PREV_DBUNDER,  dow_prev_dow_prev_in)
   277       817         2724      3.3      0.0  		DOW_DWW, DOW_DBW, DOW_DWUNDER, DOW_DBUNDER = do_dw__inputs(WW, WUNDER, BUNDER, OW_PREV_PREV[F], OUNDER_PREV, DOW_DWUNDER, \
   278       817     21899785  26805.1     12.6  											DOW_DBUNDER, OW_PREV, DOW_DWW, DOW_DBW, mem_prev_prev, x_prev, dow_prev_dow_prev_in)
   279                                           		
   280       817         2051      2.5      0.0  		DMEM_PREV_DWW, DMEM_PREV_DBW, DMEM_PREV_DWUNDER, DMEM_PREV_DBUNDER = mem_partials(DMEM_PREV_DWW, DMEM_PREV_DBW, \
   281       817          738      0.9      0.0  											DMEM_PREV_DWUNDER, DMEM_PREV_DBUNDER, DOW_DWW, \
   282       817          676      0.8      0.0  											DOW_DBW, DOW_DWUNDER, DOW_DBUNDER, OW_PREV, OUNDER_PREV, WW, BW, \
   283       817     32385329  39639.3     18.6  											WUNDER, BUNDER, x_prev, mem_prev_prev)
   284                                           	
   285                                           	# partials from read head output (OR)
   286       819     13544388  16537.7      7.8  	DOR_DWR, DOR_DBR, DOR_DWUNDER, DOR_DBUNDER = do_dw__o_prev(WR, OR_PREV[F], DOR_DWR, DOR_DBR, DOR_DWUNDER, DOR_DBUNDER, OR, dor_dor_in)
   287       819         2595      3.2      0.0  	DOR_DWR, DOR_DBR, DOR_DWUNDER, DOR_DBUNDER = do_dw__inputs(WR, WUNDER, BUNDER, OR_PREV[F], OUNDER, DOR_DWUNDER, DOR_DBUNDER, \
   288       819     21454433  26195.9     12.3  										OR, DOR_DWR, DOR_DBR, mem_prev, x, dor_dor_in)
   289                                           	
   290       819     23361838  28524.8     13.4  	DOR_DWW, DOR_DBW = do_dw__o_prev(WR, OR_PREV[F], DOR_DWW, DOR_DBW, DOR_DWUNDER, DOR_DBUNDER, OR, dor_dor_in)[:2] #?
   291       819         2389      2.9      0.0  	DOR_DWW, DOR_DBW, DOR_DWUNDER, DOR_DBUNDER = do_dw__mem_prev(WR, DOR_DWW, DOR_DBW, DOR_DWUNDER, DOR_DBUNDER, OR, \
   292       819          719      0.9      0.0  										mem_prev, DMEM_PREV_DWW, \
   293       819     18015938  21997.5     10.4  										DMEM_PREV_DBW, DMEM_PREV_DWUNDER, DMEM_PREV_DBUNDER, dor_dor_in)
   294                                           	
   295       819         2416      2.9      0.0  	return DOW_DWW, DOW_DBW, DOW_DWUNDER, DOW_DBUNDER, DMEM_PREV_DWW, DMEM_PREV_DBW, \
   296       819          864      1.1      0.0  			DMEM_PREV_DWUNDER, DMEM_PREV_DBUNDER, DOR_DWR, DOR_DBR, DOR_DWUNDER, DOR_DBUNDER, DOR_DWW, DOR_DBW

