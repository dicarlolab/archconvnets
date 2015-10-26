Timer unit: 1e-06 s

Total time: 32.8226 s
File: ntm_core.py
Function: do_dw__o_prev at line 169

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   169                                           @profile
   170                                           def do_dw__o_prev(W, o_prev, DO_DW, DO_DB, DO_DWUNDER,DO_DBUNDER, O, do_do_in):
   171      1333      2036484   1527.7      6.2  	do_in_do_prev = interpolate_softmax_do_prev(O[IN], O[IN_GATE], o_prev)
   172      1333       641379    481.2      2.0  	do_do_prev = mult_partials(do_do_in, do_in_do_prev, O[IN])
   173                                           	
   174      1333     22279068  16713.5     67.9  	DO_DW_NEW = mult_partials__layers(do_do_prev, DO_DW, o_prev)
   175      1333      2896145   2172.7      8.8  	DO_DB_NEW = mult_partials__layers(do_do_prev, DO_DB, o_prev)
   176      1333      4357850   3269.2     13.3  	DO_DWUNDER_NEW = mult_partials__layers(do_do_prev, DO_DWUNDER, o_prev)
   177      1333       610010    457.6      1.9  	DO_DBUNDER_NEW = mult_partials__layers(do_do_prev, DO_DBUNDER, o_prev)
   178                                           	
   179      1333         1671      1.3      0.0  	return DO_DW_NEW, DO_DB_NEW, DO_DWUNDER_NEW, DO_DBUNDER_NEW

