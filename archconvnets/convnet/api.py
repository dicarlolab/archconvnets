import os
import tempfile
import cPickle

from . import convnet as C
from . import util
from . import layer
from collections import OrderedDict

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


def setup_training(architecture_params, training_steps, data_provider, data_path, convnet_path, basedir):
    arch_file = os.path.join(basedir, 'arch.cfg')
    odict_to_config(architecture_params, arch_file)
    commands = []
    for tind, ts in enumerate(training_steps):
        tfile = os.path.join(basedir, 'training.cfg')
        if os.path.exists(tfile):
            nfile = nfile + '.old.%d' % tind
            shutil.move(tfile, nfile)
        odict_to_config(ts['params'], tfile)
        model_file = os.path.join(basedir, 'model_%d' % tind)
        test_range = ts['test_range']
        train_range = ts['train_range']
        test_freq = ts['test_freq']
        epochs = ts['epochs']
        if tind == 0:
            command = 'python %s --data-path=%s --save-path=%s --test-range=%s --train-range=%s --data-provider=%s --test-freq=%d --epochs=%d --model-file=%s' % (convnet_path, data_path, basedir, test_range, train_raing, data_provider, test_freq, epochs, model_file)
        else:
            prev_fn = os.path.join(basedir, 'model_%d' % (tind - 1))
            command = 'python %s -f %s --test-freq=%d --epochs=%d --model-file=%s' % (convnet_path, prev_fn, test_freq, epochs, model_file)
        commands.append(command)

    return commands
