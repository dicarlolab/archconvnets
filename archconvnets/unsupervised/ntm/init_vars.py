import numpy as np

n_shifts = 3
C = 4
M = 5
mem_length = 8
n2 = 6
n1 = 7
n_in = 3

SCALE = .6
N_FRAMES = 4

mem_previ = np.random.normal(size=(M, mem_length))

o_previ = np.random.normal(size=(C,M))
ow_previ = np.random.normal(size=(C,M))
o_content = np.random.normal(size=(C,M))
ow_content = np.random.normal(size=(C,M))

w1 = np.random.normal(size=(n1,n_in)) * SCALE
w2 = np.random.normal(size=(n2,n1)) * SCALE
w3 = np.random.normal(size=(C,n2)) * SCALE
wshift = np.random.normal(size=(C,n_in)) * SCALE

ww1 = np.random.normal(size=(n1, n_in)) * SCALE
ww2 = np.random.normal(size=(n2, n1)) * SCALE
ww3 = np.random.normal(size=(C, n2)) * SCALE
wwshift = np.random.normal(size=(C,n_in)) * SCALE

W = [w1, w2, w3, wshift]; DW = [None] * 4
WW = [ww1, ww2, ww3, wwshift]; DWW = [None] * 4

shift_out = np.random.normal(size=(C, n_shifts))
shiftw_out = np.random.normal(size=(C, n_shifts))
add_out = np.random.normal(size=(C, mem_length)) * SCALE

x = np.random.normal(size=(N_FRAMES+1, n_in,1)) * SCALE
t = np.random.normal(size=(C,mem_length))

x[0] = np.zeros_like(x[0])

do_dw3 = np.zeros((C,M,C,n2))
do_dw2 = np.zeros((C,M,n2,n1))
do_dw1 = np.zeros((C,M,n1,n_in))

dow_dww1 = np.zeros((C,M, n1,n_in))
dow_dww2 = np.zeros((C,M, n2,n1))
dow_dww3 = np.zeros((C,M, C,n2))

DO_DWi = [do_dw1, do_dw2, do_dw3]
DO_DWWi = [dow_dww1, dow_dww2, dow_dww3]

do_content_dw3 = np.zeros_like(do_dw3)
do_content_dw2 = np.zeros_like(do_dw2)
do_content_dw1 = np.zeros_like(do_dw1)

dow_content_dww1 = np.zeros_like(dow_dww1)
dow_content_dww2 = np.zeros_like(dow_dww2)
dow_content_dww3 = np.zeros_like(dow_dww3)

DO_CONTENT_DW = [do_content_dw1, do_content_dw2, do_content_dw3]
DOW_CONTENT_DWW = [dow_content_dww1, dow_content_dww2, dow_content_dww3]

dmem_prev_dww1 = np.zeros((M, mem_length, n1,n_in))
dmem_prev_dww2 = np.zeros((M, mem_length, n2,n1))
dmem_prev_dww3 = np.zeros((M, mem_length, C,n2))

DMEM_PREV_DWWi = [dmem_prev_dww1, dmem_prev_dww2, dmem_prev_dww3]

L1 = 0; L2 = 1; L3 = 2; SHIFT = 3
IN = 0; SQ = 1; F = 2
O_PREVi = [None, None, o_previ]
OW_PREVi = [np.zeros_like(ow_previ), np.zeros_like(ow_previ), ow_previ]
OW_PREV_PREVi = [None, None, np.zeros_like(ow_previ)]
GW_PREVi = [np.zeros((n1,1)), np.zeros((n2,1)), np.zeros((C,1)), np.zeros((C,1))]
