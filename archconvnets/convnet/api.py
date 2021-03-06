import re
import os
import tempfile
import cPickle
import numpy as np

from . import convnet as C
from . import util
from . import layer
from collections import OrderedDict

from yamutils import fast

CDIR = os.path.abspath(os.path.split(__file__)[0])

def unpickle(f):
    curdir = os.getcwd()
    try:
        os.chdir(CDIR)
        X = util.unpickle(f)
    except Exception as e:
        os.chdir(curdir)
        raise e 
    else:
        os.chdir(curdir)
        return X


def odict_to_config(X, savepath=None):
    mcp = layer.MyConfigParser(dict_type=OrderedDict)
    for x in X:
        mcp.add_section(x)
        for y in X[x]:
            mcp.set(x, y, str(X[x][y]))
    if savepath is not None:
        with open (savepath, 'wb') as _f:
            mcp.write(_f)
    return mcp


def configfile_to_dict(configpath):
    mcp = layer.MyConfigParser(dict_type=OrderedDict)
    mcp.read([configpath])
    return configparser_to_dict(mcp)


def configparser_to_dict(mcp):
    X = OrderedDict()
    for s in mcp.sections():
        X[s] = OrderedDict()
        for o in mcp.options(s):
            X[s][o] = mcp.get(s, o)
    return X


def setup_training(architecture_params, training_params, training_steps, data_provider,
                   data_path, convnet_path, basedir):
    arch_file = os.path.join(basedir, 'architecture.cfg')
    odict_to_config(architecture_params, arch_file)
    
    training_file = os.path.join(basedir, 'training.cfg')
    odict_to_config(training_params, training_file)
    commands = []
    for tind, ts in enumerate(training_steps):
        model_file = os.path.join(basedir, 'model_%d' % tind)
        ops = [('multiview-test', 'multiview_test', '%d'),
               ('train_range', 'train-range', '%s'),
               ('test_range', 'test-range', '%s'),
               ('test_freq', 'test-freq', '%d'),
               ('epochs', 'epochs', '%d'), 
               ('img_flip', 'img-flip', '%d'),
               ('img_rs', 'img-rs', '%d'),
               ('reset_mom', 'reset-mom', '%d'),
               ('scale_rate', 'scale-rate', '%f')]
    
        if tind == 0:
            command = 'python %s --data-path=%s --save-path=%s --data-provider=%s --model-file=%s'\
                      % (convnet_path, data_path, basedir, data_provider,  model_file)
        else:
            prev_fn = os.path.join(basedir, 'model_%d' % (tind - 1))
            command = 'python %s -f %s --model-file=%s' % (convnet_path, prev_fn, model_file)
        for optname, optstr, fmt in ops:
            if optname in ts:
                command += (' --%s=%s' % (optstr, fmt)) % ts[optname]
                
        commands.append(command)

    return commands


def get_batches(feat_dir):
    p = re.compile('data_batch_([\d]+)')
    L = os.listdir(feat_dir)
    bns = map(int, [p.match(l).groups()[0] for l in L if p.match(l)])
    bns.sort()
    return bns    


def assemble_feature_batches(feat_dir, N=None, seed=0, batch_range=None, shape_reduce=None, perm=None):
    bns = get_batches(feat_dir)
    if batch_range is not None:
        bns = bns[batch_range[0]: batch_range[1]]
    data = []
    for x in bns:
        ft = unpickle(os.path.join(feat_dir, 'data_batch_%d' % x))['data']
        if shape_reduce is not None:
            nf0, nf1 = ft.shape
            s = int(np.sqrt(nf1 / float(shape_reduce)))
            ft = ft.reshape((nf0, shape_reduce, s, s))
            ft1 = ft.mean(2).mean(2)
            ft2 = ft.sum(1).reshape((nf0, s * s))
            if N is not None:
                nf1 = ft1.shape[1]
                nf2 = ft2.shape[1]
                rng = np.random.RandomState(seed=seed)
                units1 = rng.permutation(nf1)[: N/2]
                units2 = rng.permutation(nf2)[: N - len(units1)]
                ft = np.column_stack([ft1[:, units1], ft2[:, units2]])
            else:
                ft = np.column_stack([ft1, ft2])
        elif N is not None:
            print('subsetting batch %d' % x)
            ft = ft[:, np.random.RandomState(seed=seed).permutation(ft.shape[1])[:N]]
        #data.append(unpickle(os.path.join(feat_dir, 'data_batch_%d' % x))['data'])
        data.append(ft)
    data = np.row_stack(data)
    if perm == 'random':
        _perm = np.random.RandomState(0).permutation(len(data))
        pinv = fast.perminverse(_perm)
        data = data[pinv]
    elif perm == None:
        pass
    else:
        raise ValueError("unknown perm type: %s" % perm)
    return data
