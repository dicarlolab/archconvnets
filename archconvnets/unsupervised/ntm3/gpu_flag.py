GPU = True

BATCH_SZ = 32
PANDA_PORT = 60000
GPU_IND = 3
IM_SZ = 32

N_FUTURE = 3 # how far into the future to predict
TIME_LENGTH = 3
EPOCH_LEN = N_FUTURE + TIME_LENGTH

CLASS_IMGNET = False
CLASS_CIFAR = False

IM_SZ_R = 16
N_TARGET = IM_SZ_R*IM_SZ_R*3

NO_MEM = True
