Timer unit: 1e-06 s

Total time: 13.0791 s
File: ntm_core.py
Function: do_dw__inputs at line 103

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   103                                           @profile
   104                                           def do_dw__inputs(W, WUNDER, BUNDER, o_prev, OUNDER, DO_DWUNDER, DO_DBUNDER, O, DO_DW, DO_DB, mem_prev, x, do_do_in):
   105       456       458245   1004.9      3.5  	DO_DW_NEW = copy.deepcopy(DO_DW); DO_DB_NEW = copy.deepcopy(DO_DB)
   106       456       115853    254.1      0.9  	DO_DWUNDER_NEW = copy.deepcopy(DO_DWUNDER); DO_DBUNDER_NEW = copy.deepcopy(DO_DBUNDER)
   107                                           	
   108                                           	## sharpen weights
   109       456        51366    112.6      0.4  	do_dgammarelu = dsharpen_dgamma(O[SHIFTED], O[GAMMA])
   110       456        22275     48.8      0.2  	dgammarelu_dgamma = relu_dlayer_in(O[GAMMA], thresh=1)
   111       456        22954     50.3      0.2  	do_dgamma = mult_partials(do_dgammarelu, dgammarelu_dgamma, O[GAMMA])
   112       456         3080      6.8      0.0  	DO_DB_NEW[GAMMA] += do_dgamma
   113       456        11560     25.4      0.1  	dgamma_dwgamma = linear_F_dF_nsum_g(W[GAMMA], OUNDER[F_UNDER])
   114       456         6624     14.5      0.1  	dgamma_dg3under = linear_F_dx_nsum_g(W[GAMMA], OUNDER[F_UNDER])
   115       456        65751    144.2      0.5  	DO_DW_NEW[GAMMA] += mult_partials(do_dgamma, dgamma_dwgamma, O[GAMMA])
   116       456        24013     52.7      0.2  	do_dg3under = np.squeeze(mult_partials(do_dgamma, dgamma_dg3under, O[GAMMA]))
   117                                           	
   118                                           	## shift weights
   119       456       283177    621.0      2.2  	do_dgshiftedsm = dsharpen_dw(O[SHIFTED], O[GAMMA])
   120       456        69342    152.1      0.5  	dgshiftedsm_dgshiftsm = shift_w_dshift_out_nsum(O[IN])
   121       456       129383    283.7      1.0  	do_dgshiftsm = mult_partials(do_dgshiftedsm, dgshiftedsm_dgshiftsm, O[SHARPENED])
   122       456       131921    289.3      1.0  	dgshiftsm_gshift = softmax_dlayer_in_nsum(O[SHIFT])
   123       456        61239    134.3      0.5  	do_dgshift = mult_partials(do_dgshiftsm, dgshiftsm_gshift, O[SHIFT])
   124       456         5410     11.9      0.0  	DO_DB_NEW[SHIFT] += do_dgshift
   125       456       149620    328.1      1.1  	dgshift_dwshift = linear_2d_F_dF_nsum(W[SHIFT], OUNDER[F_UNDER])
   126       456         1379      3.0      0.0  	dgshift_dg3under = linear_2d_F_dx_nsum(W[SHIFT])
   127       456       508757   1115.7      3.9  	DO_DW_NEW[SHIFT] += mult_partials(do_dgshift, dgshift_dwshift, O[SHIFT])
   128       456        40046     87.8      0.3  	do_dg3under += mult_partials(do_dgshift, dgshift_dg3under, O[SHIFT])
   129                                           	
   130                                           	## interp. gradients (wrt gin_gate)
   131       456       551564   1209.6      4.2  	do_in_dgin_gate_sig = interpolate_softmax_dinterp_gate_out(O[IN], O[IN_GATE], O[CONTENT_SM], o_prev)
   132       456        63934    140.2      0.5  	do_dgin_gate_sig = mult_partials(do_do_in, do_in_dgin_gate_sig, O[IN])
   133       456        22594     49.5      0.2  	dgin_gate_sig_dgin_gate = sigmoid_dlayer_in(O[IN_GATE])
   134       456        19611     43.0      0.1  	do_dgin_gate = mult_partials(do_dgin_gate_sig, dgin_gate_sig_dgin_gate, O[IN_GATE])
   135       456         3599      7.9      0.0  	DO_DB_NEW[IN_GATE] += do_dgin_gate
   136       456        12572     27.6      0.1  	dgin_gate_dwin = linear_F_dF_nsum_g(W[IN_GATE], OUNDER[F_UNDER])
   137       456         6640     14.6      0.1  	dgin_gate_dg3under = linear_F_dx_nsum_g(W[IN_GATE], OUNDER[F_UNDER])
   138       456        67821    148.7      0.5  	DO_DW_NEW[IN_GATE] += mult_partials(do_dgin_gate, dgin_gate_dwin, O[IN_GATE])
   139       456        25806     56.6      0.2  	do_dg3under += np.squeeze(mult_partials(do_dgin_gate, dgin_gate_dg3under, O[IN_GATE]))
   140                                           	
   141                                           	## interp. gradients (wrt o_content; key)
   142       456      1832219   4018.0     14.0  	do_do_content = do_do_content__(O, do_do_in)
   143       456      1612469   3536.1     12.3  	do_content_dgkey = cosine_sim_expand_dkeys(O[KEY], mem_prev)
   144       456       289995    636.0      2.2  	do_dgkey = mult_partials(do_do_content, do_content_dgkey, O[CONTENT])
   145       456        14381     31.5      0.1  	DO_DB_NEW[KEY] += do_dgkey
   146       456       253857    556.7      1.9  	dgkey_dwkey = linear_2d_F_dF_nsum(W[KEY], OUNDER[F_UNDER])
   147       456         1521      3.3      0.0  	dgkey_dg3under = linear_2d_F_dx_nsum(W[KEY])
   148       456      3745517   8213.9     28.6  	DO_DW_NEW[KEY] += mult_partials(do_dgkey, dgkey_dwkey, O[KEY])
   149       456        85759    188.1      0.7  	do_dg3under += mult_partials(do_dgkey, dgkey_dg3under, O[KEY])
   150                                           	
   151                                           	## interp. gradients (wrt beta)
   152       456      1594409   3496.5     12.2  	do_do_content_focused = do_do_content_focused__(O, do_do_in)
   153       456        13480     29.6      0.1  	do_content_focused_dgbeta = focus_key_dbeta_out_nsum(O[CONTENT], O[BETA])
   154       456        64064    140.5      0.5  	do_dgbeta = mult_partials(do_do_content_focused, do_content_focused_dgbeta, O[CONTENT_FOCUSED])
   155       456         4365      9.6      0.0  	DO_DB_NEW[BETA] += do_dgbeta
   156       456        12259     26.9      0.1  	dgbeta_dwbeta = linear_F_dF_nsum_g(W[BETA], OUNDER[F_UNDER])
   157       456         6748     14.8      0.1  	dgbeta_dg3under = linear_F_dx_nsum_g(W[BETA], OUNDER[F_UNDER])
   158       456        67922    149.0      0.5  	DO_DW_NEW[BETA] += mult_partials(do_dgbeta, dgbeta_dwbeta, O[BETA])
   159       456        27206     59.7      0.2  	do_dg3under += np.squeeze(mult_partials(do_dgbeta, dgbeta_dg3under, O[BETA]))
   160                                           	
   161                                           	## combine weights under gradients
   162       456       196028    429.9      1.5  	DG3UNDER_DW, DG3UNDER_DB = dunder(WUNDER, BUNDER, OUNDER, x)
   163       456       257374    564.4      2.0  	DO_DWUNDER_NEW = mult_partials__layers(do_dg3under, DG3UNDER_DW, np.squeeze(OUNDER[F_UNDER]), DO_DWUNDER_NEW)
   164       456        62596    137.3      0.5  	DO_DBUNDER_NEW = mult_partials__layers(do_dg3under, DG3UNDER_DB, np.squeeze(OUNDER[F_UNDER]), DO_DBUNDER_NEW)
   165                                           	
   166       456          826      1.8      0.0  	return DO_DW_NEW, DO_DB_NEW, DO_DWUNDER_NEW, DO_DBUNDER_NEW

