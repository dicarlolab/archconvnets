import cPickle
import os

import numpy as np
import scipy.stats as stats

import archconvnets.convnet as C

def getstats(fname):
    X = C.util.unpickle(fname)
    linds = [(1, [2, 4]), (2, [8, 22]), (3, [12, 26]), (4, [14, 28]), (5, [16, 30])]
    sval = {}
    for level, ilist in linds:
        sval[level] = {}
        for l_ind in ilist:
            layer = X['model_state']['layers'][l_ind]
            w = layer['weights'][0]
            karray = stats.kurtosis(w)
            kall = stats.kurtosis(w.ravel())
            fs = X['model_state']['layers'][l_ind]['filterSize'][0]
            ws = w.shape
            w = w.reshape((ws[0] / (fs**2), fs, fs, ws[1]))
            mat = np.row_stack([np.row_stack([w[i, j, :, :] for i in range(w.shape[0])]).T for j in range(w.shape[1])] )
            cf = np.corrcoef(mat.T)
            cft = np.corrcoef(mat)
            mat2 = np.row_stack([np.row_stack([w[i, :, :, j] for i in range(w.shape[0])]).T for j in range(w.shape[3])] )
            cf2 = np.corrcoef(mat2.T)
            cf2t = np.corrcoef(mat2)
            sval[level][l_ind] = {'karray': karray, 'kall': kall, 'corr': cf, 'corr2': cf2, 'corr_t': cft, 'corr2_t': cf2t}
    return sval
            

def compute_all_stats(dirname):
    L = np.array(os.listdir('/export/imgnet_storage_full/ConvNet_full_nofc'))
    Lf = np.array(map(float, L))
    ls = Lf.argsort()
    L = L[ls][::20]
    for l in L:
        print(l)
        s = getstats(os.path.join('/export/imgnet_storage_full/ConvNet_full_nofc', l))
        with open(os.path.join(dirname, l), 'w') as _f:
            cPickle.dump(s, _f)

