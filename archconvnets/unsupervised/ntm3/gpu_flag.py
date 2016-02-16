GPU = True

BATCH_SZ = 9
PANDA_PORT = 60000
GPU_IND = 2
IM_SZ = 32

N_FUTURE = 3 # how far into the future to predict

CLASS_IMGNET = False
CLASS_CIFAR = False

IM_SZ_R = 16
N_TARGET = IM_SZ_R*IM_SZ_R*3*N_FUTURE

DIFF = True
