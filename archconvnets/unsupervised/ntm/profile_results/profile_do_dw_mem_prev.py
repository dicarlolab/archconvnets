Timer unit: 1e-06 s

Total time: 170.173 s
File: ntm_core.py
Function: do_dw__mem_prev at line 182

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   182                                           @profile
   183                                           def do_dw__mem_prev(W, DO_DW, DO_DB, DO_DWUNDER,DO_DBUNDER, O, mem_prev, DMEM_PREV_DWW, DMEM_PREV_DBW, \
   184                                           			DMEM_PREV_DWUNDER, DMEM_PREV_DBUNDER, do_do_in):
   185      7674     30461757   3969.5     17.9  	do_do_content = do_do_content__(O, do_do_in)
   186      7674     26295144   3426.5     15.5  	do_content_dmem_prev = cosine_sim_expand_dmem(O[KEY], mem_prev)
   187      7674      2133139    278.0      1.3  	do_dmem_prev = mult_partials(do_do_content, do_content_dmem_prev, O[CONTENT])
   188                                           	
   189      7674     84464507  11006.6     49.6  	DO_DW_NEW = mult_partials__layers(do_dmem_prev, DMEM_PREV_DWW, mem_prev, DO_DW)
   190      7674     11205812   1460.2      6.6  	DO_DB_NEW = mult_partials__layers(do_dmem_prev, DMEM_PREV_DBW, mem_prev, DO_DB)
   191      7674     13426503   1749.6      7.9  	DO_DWUNDER_NEW = mult_partials__layers(do_dmem_prev, DMEM_PREV_DWUNDER, mem_prev, DO_DWUNDER)
   192      7674      2175606    283.5      1.3  	DO_DBUNDER_NEW = mult_partials__layers(do_dmem_prev, DMEM_PREV_DBUNDER, mem_prev, DO_DBUNDER)
   193                                           	
   194      7674        10340      1.3      0.0  	return DO_DW_NEW, DO_DB_NEW, DO_DWUNDER_NEW, DO_DBUNDER_NEW

