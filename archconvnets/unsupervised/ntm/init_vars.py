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
N_FRAMES = 4#3

## inputs/targets
x = np.random.normal(size=(N_FRAMES+1, n_in,1)) * SCALE
t = np.random.normal(size=(C,mem_length))

#x[0] = np.zeros_like(x[0])

## weights:
wr1 = np.random.normal(size=(n1,n_in)) * SCALE
wr2 = np.random.normal(size=(n2,n1)) * SCALE
wr3 = np.random.normal(size=(C,n2)) * SCALE
wrshift = np.random.normal(size=(C,n_shifts,n_in)) * SCALE * .5
wrkey = np.random.normal(size=(C,mem_length,n_in)) * SCALE * 1e-3

ww1 = np.random.normal(size=(n1, n_in)) * SCALE
ww2 = np.random.normal(size=(n2, n1)) * SCALE
ww3 = np.random.normal(size=(C, n2)) * SCALE
wwshift = np.random.normal(size=(C,n_shifts,n_in)) * SCALE *.5
wwkey = np.random.normal(size=(C,mem_length,n_in)) * SCALE * 1e-3

WR = [wr1, wr2, wr3, wrshift, wrkey]
WW = [ww1, ww2, ww3, wwshift, wwkey]

DWR = [None] * len(WR)
DWW = [None] * len(WW)

## address and mem partials:
DOR_DWRi = [None] * len(WR)
DMEM_PREV_DWWi = [None] * len(WW)
for layer in range(len(WR)):
	DOR_DWRi[layer] = np.zeros(np.concatenate(((C, M), WR[layer].shape)))
	DMEM_PREV_DWWi[layer] = np.zeros(np.concatenate(((M, mem_length), WR[layer].shape)))

DOW_DWWi = copy.deepcopy(DOR_DWRi)
DOR_DWWi = copy.deepcopy(DOR_DWRi)

## indices
L1 = 0; L2 = 1; L3 = 2; SHIFT = 3; KEY = 4; CONTENT = 5
IN = 6; SQ = 7; F = 8

## layer outputs/initial states:
mem_previ = np.random.normal(size=(M, mem_length))

add_out = np.random.normal(size=(C, mem_length)) * SCALE

or_previ = np.random.normal(size=(C,M))
ow_previ = np.random.normal(size=(C,M))

OR_PREVi = [None]*(len(WR) + 4)
OR_PREVi[F] = or_previ
OW_PREVi = [np.zeros((n1,1)), np.zeros((n2,1)), np.zeros((C,1)), np.zeros((C,n_shifts)),\
	np.zeros((C,mem_length)), np.zeros_like(ow_previ), np.zeros_like(ow_previ), np.zeros_like(ow_previ), ow_previ]
OW_PREV_PREVi = copy.deepcopy(OW_PREVi)
OW_PREV_PREVi[F] = np.zeros_like(OW_PREV_PREVi[F])
