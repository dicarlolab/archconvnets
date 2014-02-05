import cPickle
import os

import numpy as np
import scipy.stats as stats

import pymongo as pm
from yamutils.mongo import SONify

import archconvnets.convnet as convnet
import archconvnets.convnet.gpumodel as gpumodel

def getstats(fname, linds):
    return getstats_base(convnet.util.unpickle(fname), linds)


def getstats_from_db(query, linds, checkpoint_fs_host='localhost', 
                            checkpoint_fs_port=27017,
                            checkpoint_db_name='convnet_checkpoint_db',
                            checkpoint_fs_name='convnet_checkpoint_fs'):
    dic = gpumodel.IGPUModel.load_checkpoint_from_db(query, checkpoint_fs_host, checkpoint_fs_port, checkpoint_db_name, checkpoint_fs_name)
    return dic, getstats_base(dic, linds)


def getstats_base(X, linds):
    sval = {}
    for level, ilist in linds:
        for l_ind in ilist:
            print(l_ind)
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
            lname = X['model_state']['layers'][l_ind]['name']
            sval[lname] = {'karray': karray, 'kall': kall, 'corr': cf, 'corr2': cf2, 'corr_t': cft, 'corr2_t': cf2t}
    return sval
            

def compute_all_stats(dirname):
    L = np.array(os.listdir('/export/imgnet_storage_full/ConvNet_full_nofc'))
    Lf = np.array(map(float, L))
    ls = Lf.argsort()
    L = L[ls][::20]
    for l in L:
        print(l)
        s = getstats(os.path.join('/export/imgnet_storage_full/ConvNet_full_nofc', l), linds = [(1, [2, 4]), (2, [8, 22]), (3, [12, 26]), (4, [14, 28]), (5, [16, 30])])
        with open(os.path.join(dirname, l), 'w') as _f:
            cPickle.dump(s, _f)


def compute_all_stats_cifar_stats(dirname):

    linds = [(1, [2]), (2, [5])]

    """ 
    if not os.path.exists(os.path.join(dirname, 'color_orig')):
        os.makedirs(os.path.join(dirname, 'color_orig'))
    dirn = '/home/darren/cifar_checkpoints/color/orig/ConvNet__2014-01-15_12.13.14'
    L = np.array(os.listdir(dirn))
    Lf = np.array(map(float, L))
    ls = Lf.argsort()
    L = L[ls]
    for l in L:
        print(l)
        s = getstats(os.path.join(dirn, l), linds=linds)
        with open(os.path.join(dirname, 'color_orig', l), 'w') as _f:
            cPickle.dump(s, _f)

    if not os.path.exists(os.path.join(dirname, 'no_color_orig')):
        os.makedirs(os.path.join(dirname, 'no_color_orig'))
    dirn = '/home/darren/cifar_checkpoints/no_color/orig/ConvNet__2014-01-15_14.24.24'
    L = np.array(os.listdir(dirn))
    Lf = np.array(map(float, L))
    ls = Lf.argsort()
    L = L[ls]
    for l in L:
        print(l)
        s = getstats(os.path.join(dirn, l), linds=linds)
        with open(os.path.join(dirname, 'no_color_orig', l), 'w') as _f:
            cPickle.dump(s, _f)
    

    linds = [(1, [2]), (2, [5]), (3, [8]), (4, [10])]

    if not os.path.exists(os.path.join(dirname, 'color_conv')):
        os.makedirs(os.path.join(dirname, 'color_conv'))
    dirn = '/home/darren/cifar_checkpoints/color/conv_instead_of_local/ConvNet__2014-01-16_17.40.17'
    L = np.array(os.listdir(dirn))
    Lf = np.array(map(float, L))
    ls = Lf.argsort()
    L = L[ls]
    for l in L:
        print(l)
        s = getstats(os.path.join(dirn, l), linds=linds)
        with open(os.path.join(dirname, 'color_conv', l), 'w') as _f:
            cPickle.dump(s, _f)

    """

    linds = [(1, [2]), (2, [5]), (3, [8]), (4, [11])]

    if not os.path.exists(os.path.join(dirname, 'color_convpool')):
        os.makedirs(os.path.join(dirname, 'color_convpool'))
    dirn = '/home/darren/cifar_checkpoints/color/conv_instead_of_local_w_pooling/ConvNet__2014-01-16_17.53.33'
    L = np.array(os.listdir(dirn))
    Lf = np.array(map(float, L))
    ls = Lf.argsort()
    L = L[ls]
    for l in L:
        print(l)
        s = getstats(os.path.join(dirn, l), linds=linds)
        with open(os.path.join(dirname, 'color_convpool', l), 'w') as _f:
            cPickle.dump(s, _f)

import gridfs
def compute_all_synth_0_stats():
    N = 5
    conn = pm.Connection()
    linds = [(1, [4, 2]), (2, [8, 26]), (3, [12, 30]), (4, [14, 32]), (5, [16, 34])]
    linds = [(1, [4, 2])]
    checkpoint_db_name = 'convnet_checkpoint_db'
    checkpoint_fs_name = 'convnet_checkpoints_fs'
    #edata = {'experiment_data.experiment_id': "synthetic_training_bsize256_large_category"}
    #edata = {'experiment_data.experiment_id': "synthetic_training_bsize256",
    #          #'layer_def': '/home/yamins/archconvnets/archconvnets/convnet/ut_model_full/layer_mod2.cfg'
    #     'dp_params.dataset_name': [u'dldata.stimulus_sets.synthetic.synthetic_datasets', u'TrainingDatasetLarge']
    #    }
    #edata = {'experiment_data.experiment_id': "synthetic_training_bsize256_firstlayer"}
    #edata = {'experiment_data.experiment_id': "synthetic_training_firstlayer_large"}
    #edata = {'experiment_data.experiment_id': "synthetic_training_firstlayer_large_category"}
    edata = {}
    linds = [(1, [4, 2]), (2, [8, 26]), (3, [12, 30]), (4, [14, 32]), (5, [16, 34])]
    checkpoint_fs_name = 'reference_models'
    coll = conn[checkpoint_db_name][checkpoint_fs_name + '.files']
    N = 1
    ids = list(coll.find(edata, fields=['_id']).sort('timestamp'))[::N]
    print ids

    fs = gridfs.GridFS(conn['convnet_checkpoint_db'], 'convnet_checkpoint_filter_fs')
    for idq in ids:
        print(idq)
        dic, s = getstats_from_db(idq, linds, checkpoint_db_name=checkpoint_db_name, checkpoint_fs_name=checkpoint_fs_name)
        dic['rec']['checkpoint_db_name'] = checkpoint_db_name
        dic['rec']['checkpoint_fs_name'] = checkpoint_fs_name
        blob = cPickle.dumps(s)
        fs.put(blob, **dic['rec']) 



