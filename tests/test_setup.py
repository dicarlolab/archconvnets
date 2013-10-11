import os
from collections import OrderedDict

import archconvnets.convnet.api as api
import archconvnets.convnet.layer as layer

def test_setup_training_smoketest():

    layer_def_path = os.path.join(os.path.split(__file__)[0], '../archconvnets/convnet/cifar-layers/layers-80sec.cfg')
    layer_params_path = layer_def_path = os.path.join(os.path.split(__file__)[0], '../archconvnets/convnet/cifar-layers/layer-params-80sec.cfg')
    layerconfigs = api.configfile_to_dict(layer_def_path)
    layerparamconfigs = api.configfile_to_dict(layer_params_path)
    
    training_steps = [{'epochs': 1}, 
                      {'epochs': 2,
                       'test_range': '1-6',
                       'train_range': '3', 
                       'test_freq': 3,
                       'img_flip': 1,
                       'img_rs': 0, 
                       'reset_mom': 1,
                       'scale_rate': 4.3}]

    data_provider = 'test_dataprovider'

    data_path = 'test_datapath'

    convnet_path = 'convnet.py'
    basedir = os.path.join(os.path.split(__file__)[0], 'temp_dir_test_setup_smoketest')

    if not os.path.exists(basedir):
        os.makedirs(basedir)
    commands = api.setup_training(layerconfigs, layerparamconfigs, training_steps, data_provider,
                       data_path, convnet_path, basedir)

    assert len(commands) == 2
    layerconfigstest = api.configfile_to_dict(os.path.join(basedir, 'architecture.cfg'))
    layerparamconfigstest = api.configfile_to_dict(os.path.join(basedir, 'training.cfg'))

    assert layerconfigs == layerconfigstest
    assert layerparamconfigs == layerparamconfigstest

