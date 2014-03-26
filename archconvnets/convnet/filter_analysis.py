import cPickle
import os
import copy

import numpy as np
import scipy.stats as stats

import pymongo as pm
from yamutils.mongo import SONify

import archconvnets.convnet as convnet
import archconvnets.convnet.gpumodel as gpumodel
import sklearn.linear_model as sklinear_model

def getstats(fname, linds):
    return getstats_base(convnet.util.unpickle(fname), linds)


def getstats_from_db(query, linds, checkpoint_fs_host='localhost',
                            checkpoint_fs_port=27017,
                            checkpoint_db_name='convnet_checkpoint_db',
                            checkpoint_fs_name='convnet_checkpoint_fs'):
    dic = gpumodel.IGPUModel.load_checkpoint_from_db(query, checkpoint_fs_host, checkpoint_fs_port, checkpoint_db_name, checkpoint_fs_name)
    if linds is None:
        linds = [i for i in range(len(dic['model_state']['layers'])) if 'weights' in dic['model_state']['layers'][i]]
    return dic['rec'], getstats_base(dic, linds)


def getstats_base(X, linds):
    sval = {}
    for l_ind in linds:
        print(l_ind, X['model_state']['layers'][l_ind]['name'])
        layer = X['model_state']['layers'][l_ind]
        w = layer['weights'][0]
        karray = stats.kurtosis(w)
        kall = stats.kurtosis(w.ravel())
        cf0 = np.corrcoef(w)
        cf0t = np.corrcoef(w.T)
        wmean = w.mean(1)
        w2mean = (w**2).mean(1)
        lname = X['model_state']['layers'][l_ind]['name']
        sval[lname] = {'karray': karray, 'kall': kall, 'corr0': cf0, 'corr0_t': cf0t,
                            'wmean': wmean, 'w2mean': w2mean}

        if 'filterSize' in X['model_state']['layers'][l_ind]:
            fs = X['model_state']['layers'][l_ind]['filterSize'][0]
            ws = w.shape
            w = w.reshape((ws[0] / (fs**2), fs, fs, ws[1]))
            mat = np.row_stack([np.row_stack([w[i, j, :, :] for i in range(w.shape[0])]).T for j in range(w.shape[1])] )
            cf = np.corrcoef(mat.T)
            cft = np.corrcoef(mat)
            mat2 = np.row_stack([np.row_stack([w[i, :, :, j] for i in range(w.shape[0])]).T for j in range(w.shape[3])] )
            cf2 = np.corrcoef(mat2.T)
            cf2t = np.corrcoef(mat2)
            sval[lname].update({'corr': cf, 'corr2': cf2, 'corr_t': cft, 'corr2_t': cf2t})

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

    conn = pm.Connection()
    N = 5
    #linds = [(1, [4, 2]), (2, [8, 26]), (3, [12, 30]), (4, [14, 32]), (5, [16, 34])]
    #linds = [(1, [4, 2])]
    checkpoint_db_name = 'convnet_checkpoint_db'
    checkpoint_fs_name = 'convnet_checkpoint_fs'
    #edata = {'experiment_data.experiment_id': "synthetic_training_bsize256_large_category"}
    #edata = {'experiment_data.experiment_id': "synthetic_training_bsize256",
    #          #'layer_def': '/home/yamins/archconvnets/archconvnets/convnet/ut_model_full/layer_mod2.cfg'
    #     'dp_params.dataset_name': [u'dldata.stimulus_sets.synthetic.synthetic_datasets', u'TrainingDatasetLarge']
    #    }
    #edata = {'experiment_data.experiment_id': "synthetic_training_bsize256_firstlayer"}
    #edata = {'experiment_data.experiment_id': "synthetic_training_firstlayer_large"}
    #edata = {'experiment_data.experiment_id': "synthetic_training_firstlayer_large_category"}
    edata = {}
    #linds = [(1, [4, 2]), (2, [8, 26]), (3, [12, 30]), (4, [14, 32]), (5, [16, 34])]
    checkpoint_fs_name = 'reference_models'
    coll = conn[checkpoint_db_name][checkpoint_fs_name + '.files']
    N = 1
    ids = list(coll.find(edata, fields=['_id']).sort('timestamp'))[::N]

    fs = gridfs.GridFS(conn['convnet_checkpoint_db'], 'convnet_checkpoint_filter_fs')
    for idq in ids:
        print(idq)
        dic, s = getstats_from_db(idq, None, checkpoint_db_name=checkpoint_db_name, checkpoint_fs_name=checkpoint_fs_name)
        dic['rec']['checkpoint_db_name'] = checkpoint_db_name
        dic['rec']['checkpoint_fs_name'] = checkpoint_fs_name
        blob = cPickle.dumps(s)
        fs.put(blob, **dic['rec'])


def compute_more_stats():
    conn = pm.Connection()
    fs = gridfs.GridFS(conn['convnet_checkpoint_db'], 'convnet_checkpoint_filter_fs_new')

    qs = [({'experiment_data.experiment_id': "synthetic_training_bsize256_large_category"},
           'convnet_checkpoint_fs'),
          ({u'experiment_data.experiment_id': u'imagenet_training_reference_0'},
           'reference_models'),
          ({u'experiment_data.experiment_id': u'imagenet_training_reference_0_nofc'}, 'reference_models')]

    for q, fname in qs:
        print q, fname
        dic, s = getstats_from_db(q, None, checkpoint_fs_name=fname)
        dic['checkpoint_fs_name'] = fname
        dic['old_id'] = dic.pop('_id')
        for lname in s:
            dic1 = copy.deepcopy(dic)
            dic1['lname'] = lname
            dic1['filename'] = str(dic1['old_id']) + '_' + lname
            blob = cPickle.dumps(s[lname])
            fs.put(blob, **dic1)


def add_stds(m1, m2, s1, s2, n1, n2):
    v1 = s1**2
    v2 = s2**2
    n1 = float(n1)
    n2 = float(n2)
    a = (n1 / (n1+n2)) * (v1 + m1**2)
    b = (n2 / (n1+n2)) * (v2 + m2**2)
    c = (n1 / (n1+n2)) * m1
    d = (n2 / (n1+n2)) * m2
    t = a + b - (c + d)**2
    return np.sqrt(t)

from dldata.metrics import utils
from dldata.metrics import classifier

def compute_performance_online_batches(lname, bdir, train_batches, test_batches, mname, normalize=True, ctype='MCC'):
    train_abatches = [os.path.join(bdir, mname + '_ChallengeSynsets2013_offline_' + lname + 'a', 'data_batch_%d' % tb) for tb in train_batches]
    train_bbatches = [os.path.join(bdir, mname + '_ChallengeSynsets2013_offline_' + lname + 'b', 'data_batch_%d' % tb) for tb in train_batches]

    a_meta = cPickle.loads(open(os.path.join(bdir, mname + '_ChallengeSynsets2013_offline_' + lname + 'a', 'batches.meta')).read())
    N_a = 1000
    b_meta = cPickle.loads(open(os.path.join(bdir, mname + '_ChallengeSynsets2013_offline_' + lname + 'b', 'batches.meta')).read())
    N_b = 1000
    assert (a_meta['label_names'] == b_meta['label_names']).all()
    label_names = a_meta['label_names']
    labelset = np.arange(len(label_names))
    if ctype == 'MCC':
        cls = classifier.MaximumCorrelationClassifier2(labelset, N_a + N_b)
        cls.initialize()
    elif ctype == 'SGD':
        cls = sklinear_model.SGDClassifier()

    if normalize:
        train_mean = None
        train_std = None
        N = 0
        for a, b in zip(train_abatches, train_bbatches):
            print(a, b)
            A = cPickle.loads(open(a).read())
            A['data'] = A['data'][:, np.random.RandomState(0).permutation(A['data'].shape[1])[:N_a]]
            B = cPickle.loads(open(b).read())
            B['data'] = B['data'][:, np.random.RandomState(0).permutation(B['data'].shape[1])[:N_b]]
            F = np.column_stack([A['data'], B['data']])

            if train_mean is None:
                train_mean = F.mean(axis=0)
                train_std = np.maximum(F.std(axis=0), 1e-8)
            else:
                fr = float(N) / (N + len(F))
                m2 = F.mean(axis=0)
                train_mean = fr * train_mean + (1 - fr) * F.mean(axis=0)
                train_std = add_stds(train_mean, m2, train_std, F.std(axis=0), N, len(F))

            N += len(F)
        normalizer = lambda x: (x - train_mean) / np.maximum(train_std, 1e-8)


    for a, b in zip(train_abatches, train_bbatches):
        print(a, b)
        A = cPickle.loads(open(a).read())
        A['data'] = A['data'][:, np.random.RandomState(0).permutation(A['data'].shape[1])[:N_a]]
        B = cPickle.loads(open(b).read())
        B['data'] = B['data'][:, np.random.RandomState(0).permutation(B['data'].shape[1])[:N_b]]
        F = np.column_stack([A['data'], B['data']])
        if normalize:
            F = normalizer(F)
        if ctype == 'MCC':
            cls.partial_fit(F, A['labels'][0], labelcheck=False)
        else:
            cls.partial_fit(F, A['labels'][0], classes=labelset)


    test_abatches = [os.path.join(bdir, mname + '_ChallengeSynsets2013_offline_' + lname + 'a', 'data_batch_%d' % tb) for tb in test_batches]
    test_bbatches = [os.path.join(bdir, mname + '_ChallengeSynsets2013_offline_' + lname + 'b', 'data_batch_%d' % tb) for tb in test_batches]

    prediction = []
    actual = []
    for a, b in zip(test_abatches, test_bbatches):
        print(a, b)
        A = cPickle.loads(open(a).read())
        A['data'] = A['data'][:, np.random.RandomState(0).permutation(A['data'].shape[1])[:N_a]]
        B = cPickle.loads(open(b).read())
        B['data'] = B['data'][:, np.random.RandomState(0).permutation(B['data'].shape[1])[:N_b]]
        F = np.column_stack([A['data'], B['data']])
        if normalize:
            F = normalizer(F)
        prediction.append(cls.predict(F))
        actual.append(A['labels'][0])

    prediction = np.concatenate(prediction)
    actual = np.concatenate(actual).astype(int)

    actual = label_names[actual]
    prediction = label_names[prediction]

    split_result = classifier.get_test_result(actual, prediction, label_names, prefix='test')

    #utils.compute_classifier_metrics({}, [split_result], {}, slice(None), labelset)

    return split_result

from archconvnets.convnet import api
import imagenet.dldatasets as inet
def compute_performance(mname, lname, bdir):
    import dldata.stimulus_sets.hvm as hvm
    import dldata.stimulus_sets.synthetic.synthetic_datasets as sd
    from dldata.metrics import utils

    #Xa = api.assemble_feature_batches(os.path.join(bdir, mname + '_HvMWithDiscfade_' + lname + 'a'), N=1000)
    #Xb = api.assemble_feature_batches(os.path.join(bdir, mname + '_HvMWithDiscfade_' + lname + 'b'), N=1000)
    #X = np.column_stack([Xa, Xb])

    NS = 5
    ev = {'npc_train': 35,
    'npc_test': 5,
    'num_splits': NS,
    'npc_validate': 0,
    'metric_screen': 'classifier',
    'metric_labels': None,
    'metric_kwargs': {'model_type': 'MCC2'},
    'labelfunc': 'category',
    'train_q': {'var': ['V6']},
    'test_q': {'var': ['V6']},
    'split_by': 'obj'}

    
    #dataset = hvm.HvMWithDiscfade()
    #meta = dataset.meta
    #meta0 = meta[np.random.RandomState(0).permutation(len(meta))]
    #rec = utils.compute_metric_base(X, meta0, ev)

    #dataset = sd.TrainingDatasetLarge()
    dataset = inet.ChallengeSynsets2013_offline()
    meta = dataset.meta

    #Xa = api.assemble_feature_batches(os.path.join(bdir, mname + '_TrainingDatasetLarge_' + lname + 'a'))
    #Xb = api.assemble_feature_batches(os.path.join(bdir, mname + '_TrainingDatasetLarge_' + lname + 'b'))
    Xa = api.assemble_feature_batches(os.path.join(bdir, mname + '_ChallengeSynsets2013_offline_' + lname + 'a'), N=1000)
    Xb = api.assemble_feature_batches(os.path.join(bdir, mname + '_ChallengeSynsets2013_offline_' + lname + 'b'), N=1000)
    X = np.column_stack([Xa, Xb])

    meta0 = meta[np.random.RandomState(0).permutation(len(meta))][: X.shape[0]]


    #NS = 5
    #ev = {'npc_train': 31,
    #'npc_test': 5,
    #'num_splits': NS,
    #'npc_validate': 0,
    #'metric_screen': 'classifier',
    #'metric_labels': None,
    #'metric_kwargs': {'model_type': 'MCC2'},
    #'labelfunc': 'category',
    #'train_q': {},
    #'test_q': {},
    #'split_by': 'obj'}
    #rec_t = utils.compute_metric_base(X, meta0, ev)

    ev = {'npc_train': 150, 
       	  'npc_test': 50,
          'num_splits': NS,
	  'npc_validate': 0,
	  'metric_screen' : 'classifier', #
	  'metric_labels': None,
	  'metric_kwargs': {'model_type': 'MCC2'},
	  'labelfunc': 'synset',
	  'train_q': {},
	  'test_q': {},
	  'split_by': 'synset'}

    rec_c = utils.compute_metric_base(X, meta0, ev)

    return {
            #'rec_hvm': rec, 
            #'rec_training': rec_t
             'rec_challenge': rec_c}


def do_train(outdir, bdir, post=''):
    #model_names = ["imagenet_trained", "synthetic_category_trained", "rosch_trained"]
    model_names = ["imagenet_trained", "rosch_trained"]
    layer_names = ['conv1_1','conv2_4', 'conv3_7', 'conv4_8','conv5_9', 'pool3_11', 'fc1_12', 'rnorm4_13','fc2_14', 'rnorm5_15']
    for mname in model_names[:]:
        for lname in layer_names:
            print(mname, lname)
            res = compute_performance(mname, lname, bdir)
            fname = os.path.join(outdir, '%s_%s%s.pkl' %(mname, lname, post))
            with open(fname, 'w') as _f:
                cPickle.dump(res, _f)

    #model_names = ["imagenet_trained_nofc"]
    #layer_names = ['conv1_1', 'conv2_4','conv3_7','conv4_8','conv5_9', 'pool3_11']
    #for mname in model_names:
    #    for lname in layer_names:
    #        print(mname, lname)
    #        res = compute_performance(mname, lname, bdir)
    #        fname = os.path.join(outdir, '%s_%s.pkl' %(mname, lname))
    #        with open(fname, 'w') as _f:
    #            cPickle.dump(res, _f)
