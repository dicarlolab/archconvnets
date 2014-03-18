import os
import copy
import numpy.random as nr
import json
import cPickle
import hashlib
import tempfile

import hyperopt
try:
    from hyperopt.pyll import scope
except ImportError:
    print 'Trying standalone pyll'
    from pyll import scope

from ..convnet.gpumodel import IGPUModel
from ..convnet.convnet import ConvNet
from ..convnet.api import odict_to_config
from ..convnet.layer import LayerParsingError

from . import imgnet_params_new_no_bias
from .hyperopt_helpers import suggest_multiple_from_name


def imgnet_random_experiment_new_no_bias(experiment_id):
    dbname = 'imgnet_predictions_random_experiment_new_no_bias'
    host = 'localhost'
    port = 6667
    bandit = 'imgnet_prediction_bandit_new_no_bias'
    bandit_kwargdict = {'param_args': {'num_layers': 1}, 'experiment_id': experiment_id}
    exp = imgnet_random_experiment(dbname, host, port, bandit, bandit_kwargdict)
    return exp


def imgnet_random_experiment(dbname, host, port, bandit, bandit_kwargdict):
    num = 1
    bandit_algo_names = ['hyperopt.Random'] * num
    bandit_names = ['archconvnets.hyperopt.imgnet_prediction.%s' % bandit] * num
    #ek = hashlib.sha1(cPickle.dumps(bandit_kwargdict)).hexdigest()
    exp_keys = ['imgnet_prediction_random_%s_%s_%i' % (bandit, bandit_kwargdict['experiment_id'], i) for i in range(num)]
    bandit_args_list = [(bandit_kwargdict,) for i in range(num)]
    bandit_kwargs_list = [{} for i in range(num)]
    return suggest_multiple_from_name(dbname=dbname,
                               host=host,
                               port=port,
                               bandit_algo_names=bandit_algo_names,
                               bandit_names=bandit_names,
                               exp_keys=exp_keys,
                               N=None,
                               bandit_args_list=bandit_args_list,
                               bandit_kwargs_list=bandit_kwargs_list,
                               bandit_algo_args_list=[() for _i in range(num)],
                               bandit_algo_kwargs_list=[{} for _i in range(num)])


bandit_exceptions = [
            (
                lambda e:
                    isinstance(e, LayerParsingError)
                ,
                lambda e: {
                    'loss': float(1.0),
                    'status': hyperopt.STATUS_FAIL,
                    'failure': repr(e)
                }
            ),
        ]


@hyperopt.base.as_bandit(exceptions=bandit_exceptions)
def imgnet_prediction_bandit_new_no_bias(argdict):
    template = imgnet_params_new_no_bias.template_func(argdict['param_args'])
    interpreted_template = scope.config_interpret(template)
    return scope.imgnet_prediction_bandit_evaluate(interpreted_template, argdict)


@scope.define
def config_interpret(template):
    return imgnet_params_new_no_bias.config_interpretation(template)


@scope.define
def imgnet_prediction_bandit_evaluate(config, kwargs, features=None):
    _, layer_fname = tempfile.mkstemp()
    odict_to_config(config['layer_def'], savepath=layer_fname)
    _, layer_param_fname = tempfile.mkstemp()
    odict_to_config(config['learning_params'], savepath=layer_param_fname)

    exp_id = kwargs['experiment_id']
    fs_name = 'imgnet_prediction'
    config_str = json.dumps(config)
    config_id = hashlib.sha1(config_str).hexdigest()
    exp_str = json.dumps({"experiment_id": exp_id,
                          "config_id": config_id})

    op = ConvNet.get_options_parser()
    oppdict = [('--save-db', '1'),
               ('--crop', '9'),
               ('--train-range', '0-4'),
               ('--test-range', '5'),
               ('--layer-def', layer_fname),
               ('--conserve-mem', '1'),
               ('--layer-params', layer_param_fname),
               ('--data-provider', 'general-cropped'),
               ('--data-path', '/export/storage/imgnet_256batchsz_138px'),
               ('--dp-params', '{"perm_type": "random", "perm_seed": 0, "preproc": {"normalize": false, "dtype": "float32", "resize_to": [138, 138], "mode": "RGB", "crop": null, "mask": null}, "batch_size": 256, "meta_attribute": "synset", "dataset_name": ["imagenet.dldatasets", "ChallengeSynsets2013_offline"]}'),
               ('--test-freq', '100'),
               ('--saving-freq', '0'),
               ('--checkpoint-fs-port', '6666'),
               ('--epochs', '10'),
               ('--img-size', '138'),
               ('--experiment-data', exp_str),
               ('--checkpoint-db-name', 'imgnet_prediction'),
               ("--checkpoint-fs-name", fs_name)]
    gpu_num = os.environ.get('BANDIT_GPU', None)
    if gpu_num is not None:
        oppdict.append(('--gpu', gpu_num))

    op, load_dic = IGPUModel.parse_options(op, input_opts=dict(oppdict), ignore_argv=True)
    nr.seed(0)
    model = ConvNet(op, load_dic)
    try:
        model.start()
    except SystemExit, e:
        if not e.code == 0:
            raise e

    cpt = IGPUModel.load_checkpoint_from_db({"experiment_data.experiment_id":exp_id, "experiment_data.config_id": config_id}, checkpoint_fs_host='localhost', checkpoint_fs_port=6666, checkpoint_db_name='imgnet_prediction', checkpoint_fs_name=fs_name, only_rec=True)
    rec = cpt['rec']
    rec['kwargs'] = kwargs
    rec['spec'] = config
    rec['loss'] = rec['test_outputs'][0]['logprob'][0]
    rec['status'] = 'ok'

    return rec



#cmd_tmpl  = """python convnet.py --crop=%d --test-range=0-4 --train-range=5 --layer-def=%s --layer-params=%s --data-provider=general-cropped  --dp-params='{"preproc": {"normalize": false, "dtype": "float32", "mask": null, "crop": null, "resize_to": [32, 32], "mode": "RGB"}, "batch_size": 10000, "meta_attribute": "category", "dataset_name":["dldata.stimulus_sets.cifar10", "Cifar10"]}' --test-freq=%d --epochs=%d --save-db=1 --img-size=32 --experiment-data='{"experiment_id":"%s", "config":%s, "config_id":"%s"}' --checkpoint-fs-name=%s"""  % (crop, layer_fname, layer_param_fname, tf, epochs, exp_id, config_str, config_id, fs_name)
    #retcode = subprocess.call(cmd_tmpl, shell=True)
