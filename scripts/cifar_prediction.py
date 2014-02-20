import hashlib
import subprocess

import hyperopt
try:
    from hyperopt.pyll import scope
except ImportError:
    print 'Trying standalone pyll'
    from pyll import scope


import archconvnets.convnet.gpumodel as gpumodel

from . import convnet_params

@hyperopt.base.as_bandit(exceptions=[])
def convnet_bandit(argdict):
    template = convnet_params.template_func(argdict['param_args'])
    return scope.convnet_bandit_evaluate(template, argdict)


@scope.define
def convnet_bandit_evaluate(config, kwargs, features=None):
    _, layer_fname = tempfile.mkstemp()
    odict_to_config(config['layer_def'], savepath=layer_fname)
    _, layer_param_fname = tempfile.mkstemp()
    odict_to_config(config['learning_params'], savepath=layer_param_fname)

    tf = 20*6 / 20
    epochs = 20
    exp_id = kwargs['experiment_id']

    dpath = dset.home('convnet_batches')
    crop = 4
    fs_name = 'cifar_prediction'
    config_str = json.dumps(config)
    config_id = hashlib.sha1(config_str).hexdigest()

    cmd_tmpl  = """python convnet.py --data-path=%s --crop=%d --test-range=1-5 --train-range=6 --layer-def=%s --layer-params=%s --data-provider=general-cropped  --dp-params='{"preproc": {"normalize": false, "dtype": "float32", "mask": null, "crop": null, "resize_to": [32, 32], "mode": "RGB"}, "batch_size": 10000, "meta_attribute": "category", "dataset_name":["dldata.stimulus_sets.cifar10", "Cifar10"]}' --test-freq=%d --epochs=%d --save-db=1 --img-size=32 --experiment-data='{"experiment_id":"%s", "config":%s, "config_id":%s}' --checkpoint-fs-name=%s"""  % (dpath, crop, layer_fname, layer_param_fname, tf, epochs, exp_id, config_str, config_id, fs_name)

    retcode = subprocess.call(cmd_tmpl, shell=True)

    cpt = gpumodel.load_checkpoint_from_db({"experiment_id":exp_id, "config_id": config_id},
                                            checkpoint_fs_name=fs_name,
                                            only_rec=True)
    rec = cpt['rec']
    rec['kwargs'] = kwargs
    rec['loss'] = rec['test_outputs'][0]['logprob'][0]
    rec['status'] = 'ok'

    return rec
