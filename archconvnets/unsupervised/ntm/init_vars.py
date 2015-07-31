import numpy as np
import copy

n_shifts = 3
C = 4
M = 5
mem_length = 8
n2 = 6
n1 = 7
n_in = 3
n_head_in = 9
n1_under = 10
n2_under = 11

SCALE = .6
N_FRAMES = 4
SCALE_UNDER = .4

## inputs/targets
x = np.random.normal(size=(N_FRAMES+1, n_in,1)) * SCALE
t = np.random.normal(size=(C,mem_length))

## under weights:
w1 = np.random.normal(size=(n1_under, n_in)) * SCALE_UNDER
w2 = np.random.normal(size=(n2_under, n1_under)) * SCALE_UNDER
w3 = np.random.normal(size=(n_head_in, n2_under)) * SCALE_UNDER

## head weights:
wr1 = np.random.normal(size=(n1,n_head_in)) * SCALE
wr2 = np.random.normal(size=(n2,n1)) * SCALE
wr3 = np.random.normal(size=(C,n2)) * SCALE
wrshift = np.random.normal(size=(C,n_shifts,n_head_in)) * SCALE * .5
wrkey = np.random.normal(size=(C,mem_length,n_head_in)) * SCALE * 1e-3

ww1 = np.random.normal(size=(n1, n_head_in)) * SCALE
ww2 = np.random.normal(size=(n2, n1)) * SCALE
ww3 = np.random.normal(size=(C, n2)) * SCALE
wwshift = np.random.normal(size=(C,n_shifts,n_head_in)) * SCALE *.5
wwkey = np.random.normal(size=(C,mem_length,n_head_in)) * SCALE * 1e-3

wadd = np.random.normal(size=(C, mem_length, n_head_in)) * SCALE

WUNDER = [w1, w2, w3]
WR = [wr1, wr2, wr3, wrshift, wrkey]
WW = [ww1, ww2, ww3, wwshift, wwkey, wadd]

DUNDER = [None] * len(WUNDER)
DWR = [None] * len(WR)
DWW = [None] * len(WW)

## address and mem partials:
DOR_DWUNDERi = [None] * len(WUNDER)
DOR_DWRi = [None] * len(WR)
DOR_DWWi = [None] * len(WW)

DMEM_PREV_DWWi = [None] * len(WW)
DMEM_PREV_DWUNDERi = [None] * len(WUNDER)

for layer in range(len(WR)):
	DOR_DWRi[layer] = np.zeros(np.concatenate(((C, M), WR[layer].shape)))

for layer in range(len(WUNDER)):
	DOR_DWUNDERi[layer] = np.zeros(np.concatenate(((C, M), WUNDER[layer].shape)))
	DMEM_PREV_DWUNDERi[layer] = np.zeros(np.concatenate(((M, mem_length), WUNDER[layer].shape)))

for layer in range(len(WW)):
	DOR_DWWi[layer] = np.zeros(np.concatenate(((C, M), WW[layer].shape)))
	DMEM_PREV_DWWi[layer] = np.zeros(np.concatenate(((M, mem_length), WW[layer].shape)))

DOW_DWWi = copy.deepcopy(DOR_DWWi)
DOW_DWUNDERi = copy.deepcopy(DOR_DWUNDERi)

## indices
L1_UNDER = 0; L2_UNDER = 1; F_UNDER = 2

L1 = 0; L2 = 1; L3 = 2; SHIFT = 3; KEY = 4; ADD = 5
CONTENT = 6; IN = 7; SQ = 8; F = 9

## layer outputs/initial states:
mem_previ = np.random.normal(size=(M, mem_length))

or_previ = np.random.normal(size=(C,M))
ow_previ = np.random.normal(size=(C,M))

OR_PREVi = [None]*(len(WR) + 5)
OR_PREVi[F] = or_previ
OW_PREVi = [np.zeros((n1,1)), np.zeros((n2,1)), np.zeros((C,1)), np.zeros((C,n_shifts)), np.zeros((C,mem_length)),\
	np.zeros((C,mem_length)), np.zeros_like(ow_previ), np.zeros_like(ow_previ), np.zeros_like(ow_previ), ow_previ]
OW_PREV_PREVi = copy.deepcopy(OW_PREVi)
OW_PREV_PREVi[F] = np.zeros_like(OW_PREV_PREVi[F])
OUNDER_PREVi = np.zeros((n_head_in, 1))