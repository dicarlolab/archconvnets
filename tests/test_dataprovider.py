import os
import numpy as np

import dldata.stimulus_sets.hvm as hvm
import imagenet

import archconvnets.dataprovider as dp
import archconvnets.convnet.api as api



def test_dataprovider_hvm():
    dataset = hvm.HvMWithDiscfade()
    imgs = dataset.get_images('float32', {'size': (128, 128), 'global_normalize': False})
    metadata = dataset.meta['category']
    provider = dp.Dldata2ConvnetProviderBase(imgs, metadata, 200)

    assert provider.get_data_dims() == 128 * 128, provider.get_data_dims()
    assert provider.batch_range == range(1, 30), provider.batch_range

    X = provider.get_next_batch()
    X1 = provider.get_next_batch()

    assert X[0] == X1[0] == 1
    assert X[1] == 1
    assert X1[1] == 2

    assert X[2][0].shape == X1[2][0].shape == (16384, 200)


def test_dataprovider_imagenet():
    dataset = imagenet.dldatasets.PixelHardSynsets2013ChallengeTop25Screenset()
    imgs = dataset.get_images(dataset.default_preproc)
    metadata = dataset.meta['synset']
    provider = dp.Dldata2ConvnetProviderBase(imgs, metadata, 100)

    assert provider.get_data_dims() == 256 * 256 * 3, provider.get_data_dims()

    X = provider.get_next_batch()
    X1 = provider.get_next_batch()

    assert X[0] == X1[0] == 1
    assert X[1] == 1
    assert X1[1] == 2

    assert X[2][0].shape == X1[2][0].shape == (256 * 256 * 3, 100)



def test_dataprovider_hvm_allbatches():
    dataset = hvm.HvMWithDiscfade()
    imgs = dataset.get_images('float32', {'size': (128, 128), 'global_normalize': False})
    metadata = dataset.meta['category']
    provider = dp.Dldata2ConvnetProviderBase(imgs, metadata, 200, batch_range=[1, 15])

    for i in range(3):
        X = provider.get_next_batch()

    assert X[0] == 2, 'epoch should be %d but is %d' % (2, X[0])
    assert X[1] == 1, 'batch_num should be %d but is %d' % (1, X[1])


def test_cifar10():
    data_path = os.environ['CIFAR10_PATH']
    dirn = os.path.abspath(os.path.split(__file__ )[0])
    layer_def_path = os.path.join(dirn, '../archconvnets/convnet/cifar-layers/layers-80sec.cfg')
    layer_params_path = os.path.join(dirn, '../archconvnets/convnet/cifar-layers/layer-params-80sec.cfg')
    save_path = os.path.join(dirn, 'temp_cifar10')
    convnet_path = os.path.join(dirn, '../archconvnets/convnet/convnet.py')
    mfile1 = 'model1'
    command1 = "python %s --data-path=%s --save-path=%s --test-range=6 --train-range=1-5 --layer-def=%s --layer-params=%s --data-provider=cifartest --test-freq=5 --epochs=1 --random-seed=0 --model-file=%s" % (convnet_path, data_path, save_path, layer_def_path, layer_params_path, mfile1)

    mfile2 = 'model2'
    command2 = "python %s --data-path=%s --save-path=%s --test-range=6 --train-range=1-5 --layer-def=%s --layer-params=%s --data-provider=cifar --test-freq=5 --epochs=1 --random-seed=0 --model-file=%s" % (convnet_path,data_path, save_path, layer_def_path, layer_params_path, mfile2)

    e = os.system(command1)
    e1 = os.system(command2)
    print 'trying unpickle from'
    print os.getcwd()
    A = api.unpickle(os.path.join(save_path, mfile1, '1.5'))
    B = api.unpickle(os.path.join(save_path, mfile2, '1.5'))
    a = A['model_state']['layers'][-3]['weights']
    b = B['model_state']['layers'][-3]['weights']

    assert A['op'].options['dp_type'].value == 'cifartest'
    assert B['op'].options['dp_type'].value == 'cifar'

    assert np.allclose(a, b)

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

    A = api.unpickle(os.path.join(save_path, mfile1, '1.5'))

    test_classification_error = A['model_state']['test_outputs'][0][0]['logprob'][1]
    assert test_classification_error < .3


def test_unpickle():
    print(api.CDIR)
    api.unpickle('/home/yamins/archconvnets/tests/temp_cifar10/model1/1.5')
