import numpy as np
import copy

n_shifts = 3
C = 4
M = 5
mem_length = 8
n2 = 6
n1 = 7
n_in = 3

SCALE = .6
N_FRAMES = 3

## inputs/targets
x = np.random.normal(size=(N_FRAMES+1, n_in,1)) * SCALE
t = np.random.normal(size=(C,mem_length))

x[0] = np.zeros_like(x[0])

## weights:
w1 = np.random.normal(size=(n1,n_in)) * SCALE
w2 = np.random.normal(size=(n2,n1)) * SCALE
w3 = np.random.normal(size=(C,n2)) * SCALE
wshift = np.random.normal(size=(C,n_shifts,n_in)) * SCALE * .5

ww1 = np.random.normal(size=(n1, n_in)) * SCALE
ww2 = np.random.normal(size=(n2, n1)) * SCALE
ww3 = np.random.normal(size=(C, n2)) * SCALE
wwshift = np.random.normal(size=(C,n_shifts,n_in)) * SCALE *.5

W = [w1, w2, w3, wshift]; DW = [None] * 4
WW = [ww1, ww2, ww3, wwshift]; DWW = [None] * 4

## address partials:
do_dw1 = np.zeros((C,M,n1,n_in))
do_dw2 = np.zeros((C,M,n2,n1))
do_dw3 = np.zeros((C,M,C,n2))
do_dwshift = np.zeros((C,M,C,3,n_in))

DO_DWi = [do_dw1, do_dw2, do_dw3, do_dwshift]
DO_DWWi = copy.deepcopy(DO_DWi)

DO_CONTENT_DW = [np.zeros_like(do_dw1), np.zeros_like(do_dw2), np.zeros_like(do_dw3)]
DOW_CONTENT_DWW = copy.deepcopy(DO_CONTENT_DW)

### mem partials:
dmem_prev_dww1 = np.zeros((M, mem_length, n1,n_in))
dmem_prev_dww2 = np.zeros((M, mem_length, n2,n1))
dmem_prev_dww3 = np.zeros((M, mem_length, C,n2))
dmem_prev_dwshift = np.zeros((M, mem_length, C,n_shifts,n_in))

DMEM_PREV_DWWi = [dmem_prev_dww1, dmem_prev_dww2, dmem_prev_dww3, dmem_prev_dwshift]

## indices
L1 = 0; L2 = 1; L3 = 2; SHIFT = 3
IN = 4; SQ = 5; F = 6

## layer outputs/initial states:
mem_previ = np.random.normal(size=(M, mem_length))

add_out = np.random.normal(size=(C, mem_length)) * SCALE

o_previ = np.random.normal(size=(C,M))
ow_previ = np.random.normal(size=(C,M))
o_content = np.random.normal(size=(C,M))
ow_content = np.random.normal(size=(C,M))

G_PREVi = [None]*7 
G_PREVi[F] = o_previ
GW_PREVi = [np.zeros((n1,1)), np.zeros((n2,1)), np.zeros((C,1)), np.zeros((C,n_shifts)),\
	np.zeros_like(ow_previ), np.zeros_like(ow_previ), ow_previ]
GW_PREV_PREVi = copy.deepcopy(GW_PREVi)
GW_PREV_PREVi[F] = np.zeros_like(GW_PREV_PREVi[F])