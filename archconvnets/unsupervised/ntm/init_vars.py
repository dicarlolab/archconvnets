import numpy as np
import copy

n_shifts = 3
C = 4
M = 5
mem_length = 8
n_in = 3
n_head_in = 9
n1_under = 10
n2_under = 11

SCALE = .6
N_FRAMES = 4
SCALE_UNDER = .425

## indices
L1_UNDER = 0; L2_UNDER = 1; F_UNDER = 2

IN_GATE = 0; SHIFT = 1; KEY = 2; ADD = 3
CONTENT = 4; IN = 5; SQ = 6; F = 7

## inputs/targets
x = np.random.normal(size=(N_FRAMES+1, n_in,1)) * SCALE
t = np.random.normal(size=(C,mem_length))

## under weights:
w1 = np.random.normal(size=(n1_under, n_in)) * SCALE_UNDER
w2 = np.random.normal(size=(n2_under, n1_under)) * SCALE_UNDER
w3 = np.random.normal(size=(n_head_in, n2_under)) * SCALE_UNDER

WUNDER = [w1, w2, w3]
OUNDER_PREVi = np.zeros((n_head_in, 1))

## head weights:
OR_PREVi = [None] * (3+5); OW_PREVi = [None] * (4+5) # prev states
OR_SHAPES = [None] * (3+5); OW_SHAPES = [None] * (4+5) # prev state shapes
WR_SHAPES = [None] * 3; WW_SHAPES = [None] * 4 # weight shapes
WR = [None] * 3; WW = [None] * 4 # weights

# in
WR_SHAPES[IN_GATE] = (C, n_head_in)
WW_SHAPES[IN_GATE] = (C, n_head_in)

OR_SHAPES[IN_GATE] = (C, 1)
OW_SHAPES[IN_GATE] = (C, 1)

# shift
WR_SHAPES[SHIFT] = (C, n_shifts, n_head_in)
WW_SHAPES[SHIFT] = (C, n_shifts, n_head_in)

OR_SHAPES[SHIFT] = (C, n_shifts)
OW_SHAPES[SHIFT] = (C, n_shifts)

# key
WR_SHAPES[KEY] = (C, mem_length, n_head_in)
WW_SHAPES[KEY] = (C, mem_length, n_head_in)

OR_SHAPES[KEY] = (C, mem_length)
OW_SHAPES[KEY] = (C, mem_length)

# add
WW_SHAPES[ADD] = (C, mem_length, n_head_in)
OW_SHAPES[ADD] = (C, mem_length)

# init weights
for layer in range(len(WR_SHAPES)):
	WR[layer] = np.random.normal(size = WR_SHAPES[layer]) * SCALE
	WW[layer] = np.random.normal(size = WW_SHAPES[layer]) * SCALE
	
	OR_PREVi[layer] = np.zeros(OR_SHAPES[layer])
	OW_PREVi[layer] = np.zeros(OW_SHAPES[layer])

WW[ADD] = np.random.normal(size = WW_SHAPES[ADD])
OW_PREVi[ADD] = np.zeros(WW_SHAPES[ADD])
	
###

OR_PREVi[F] = np.random.normal(size=(C,M))

OW_PREVi[IN] = np.zeros_like(OR_PREVi[F])
OW_PREVi[SQ] = np.zeros_like(OR_PREVi[F])
OW_PREVi[F] = np.random.normal(size=(C,M))

OW_PREV_PREVi = copy.deepcopy(OW_PREVi)
OW_PREV_PREVi[F] = np.zeros_like(OW_PREV_PREVi[F])
###


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
	DOR_DWRi[layer] = np.zeros(np.concatenate(((C, M), WR_SHAPES[layer])))

for layer in range(len(WUNDER)):
	DOR_DWUNDERi[layer] = np.zeros(np.concatenate(((C, M), WUNDER[layer].shape)))
	DMEM_PREV_DWUNDERi[layer] = np.zeros(np.concatenate(((M, mem_length), WUNDER[layer].shape)))

for layer in range(len(WW)):
	DOR_DWWi[layer] = np.zeros(np.concatenate(((C, M), WW_SHAPES[layer])))
	DMEM_PREV_DWWi[layer] = np.zeros(np.concatenate(((M, mem_length), WW_SHAPES[layer])))

DOW_DWWi = copy.deepcopy(DOR_DWWi)
DOW_DWUNDERi = copy.deepcopy(DOR_DWUNDERi)

## layer outputs/initial states:
mem_previ = np.random.normal(size=(M, mem_length))
