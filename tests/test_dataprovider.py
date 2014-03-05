import os
import numpy as np
from nose.plugins.attrib import attr

import dldata.stimulus_sets.hvm as hvm
import imagenet

import archconvnets.dataprovider as dp
import archconvnets.convnet.api as api



@attr('slow')
def test_80sec_performance():
    data_path = os.environ['CIFAR10_PATH']
    dirn = os.path.abspath(os.path.split(__file__ )[0])
    layer_def_path_ours = os.path.join(dirn, '../archconvnets/convnet/cifar-layers/layers-80sec.cfg')
    layer_def_path_theirs = os.path.join(dirn, '../dropnn-release/drop-nn/cifar-layers/layers-80sec.cfg')

    layer_params_path_ours = os.path.join(dirn, '../archconvnets/convnet/cifar-layers/layer-params-80sec.cfg')
    layer_params_path_theirs = os.path.join(dirn, '../dropnn-release/drop-nn/cifar-layers/layer-params-80sec.cfg')
    save_path = os.path.join(dirn, 'temp_cifar10')
    convnet_path_ours= os.path.join(dirn, '../archconvnets/convnet/convnet.py')
    convnet_path_theirs  = os.path.join(dirn, '../dropnn-release/drop-nn/convnet.py')
    mfile1 = 'model1'
    command1 = "python %s --data-path=%s --save-path=%s --test-range=6 --train-range=1-5 --layer-def=%s --layer-params=%s --data-provider=cifartest --test-freq=50 --epochs=10 --random-seed=0 --model-file=%s" % (convnet_path_ours, data_path, save_path, layer_def_path_ours, layer_params_path_ours, mfile1)


    e = os.system(command1)

    A = api.unpickle(os.path.join(save_path, mfile1, 'model1'))

    test_classification_error = A['model_state']['test_outputs'][0][0]['logprob'][1]
    assert test_classification_error < .3




