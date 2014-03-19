import os
import collections
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

from . import cifar_params
from . import cifar_params_intermediate
from . import cifar_params_intermediate2
from . import cifar_params_intermediate3
from . import cifar_params_new
from .hyperopt_helpers import suggest_multiple_from_name


def cifar_random_experiment0(experiment_id):
    dbname = 'cifar_predictions_random_experiment0'
    host = 'localhost'
    port = 22334
    bandit = 'cifar_prediction_bandit'
    bandit_kwargdict = {'param_args': {}, 'experiment_id': experiment_id}
    exp = cifar_random_experiment(dbname, host, port, bandit, bandit_kwargdict)
    return exp


def cifar_random_experiment_intermediate(experiment_id):
    dbname = 'cifar_predictions_random_experiment_intermediate'
    host = 'localhost'
    port = 22334
    bandit = 'cifar_prediction_bandit_intermediate'
    bandit_kwargdict = {'param_args': {}, 'experiment_id': experiment_id}
    exp = cifar_random_experiment(dbname, host, port, bandit, bandit_kwargdict)
    return exp


def cifar_random_experiment_intermediate2(experiment_id):
    dbname = 'cifar_predictions_random_experiment_intermediate2'
    host = 'localhost'
    port = 22334
    bandit = 'cifar_prediction_bandit_intermediate2'
    bandit_kwargdict = {'param_args': {}, 'experiment_id': experiment_id}
    exp = cifar_random_experiment(dbname, host, port, bandit, bandit_kwargdict)
    return exp
    
    
def cifar_random_experiment_intermediate3(experiment_id):
    dbname = 'cifar_predictions_random_experiment_intermediate3'
    host = 'localhost'
    port = 22334
    bandit = 'cifar_prediction_bandit_intermediate3'
    bandit_kwargdict = {'param_args': {}, 'experiment_id': experiment_id}
    exp = cifar_random_experiment(dbname, host, port, bandit, bandit_kwargdict)
    return exp
    

def cifar_tpe_experiment_intermediate3(experiment_id):
    dbname = 'cifar_predictions_tpe_experiment_intermediate3'
    host = 'localhost'
    port = 22334
    bandit = 'cifar_prediction_bandit_intermediate3'
    bandit_kwargdict = {'param_args': {}, 'experiment_id': experiment_id,
                        'epochs_round0': 70, 'epochs_round1': 75,
                        'test_freq': 25}
    exp = cifar_tpe_experiment(dbname, host, port, bandit, bandit_kwargdict,
                              num=1,
                              gamma=0.25,
                              n_startup_jobs=200)
    return exp


def cifar_random_experiment_new(experiment_id):
    dbname = 'cifar_predictions_random_experiment_new'
    host = 'localhost'
    port = 22334
    bandit = 'cifar_prediction_bandit_new'
    bandit_kwargdict = {'param_args': {'num_layers': 10}, 'experiment_id': experiment_id}
    exp = cifar_random_experiment(dbname, host, port, bandit, bandit_kwargdict)
    return exp


def cifar_random_experiment(dbname, host, port, bandit, bandit_kwargdict):
    num = 1
    bandit_algo_names = ['hyperopt.Random'] * num
    bandit_names = ['archconvnets.hyperopt.cifar_prediction.%s' % bandit] * num
    #ek = hashlib.sha1(cPickle.dumps(bandit_kwargdict)).hexdigest()
    exp_keys = ['cifar_prediction_random_%s_%s_%i' % (bandit, bandit_kwargdict['experiment_id'], i) for i in range(num)]
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


def cifar_tpe_experiment(dbname, host, port, bandit, bandit_kwargdict,
                           num,
                           gamma,
                           n_startup_jobs):
    bandit_algo_names = ['hyperopt.TreeParzenEstimator'] * num
    bandit_names = ['archconvnets.hyperopt.cifar_prediction.%s' % bandit] * num
    #ek = hashlib.sha1(cPickle.dumps(bandit_kwargdict)).hexdigest()
    exp_keys = ['cifar_prediction_tpe_%s_%s_%i' % (bandit, bandit_kwargdict['experiment_id'], i) for i in range(num)]
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
                               bandit_kwargs_list=[{} for _i in range(num)],
                               bandit_algo_args_list=[() for _i in range(num)],
                               bandit_algo_kwargs_list=[{'gamma':gamma,
                    'n_startup_jobs': n_startup_jobs} for _i in range(num)])


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
def cifar_prediction_bandit(argdict):
    template = cifar_params.template_func(argdict['param_args'])
    return scope.cifar_prediction_bandit_evaluate(template, argdict)


@hyperopt.base.as_bandit(exceptions=bandit_exceptions)
def cifar_prediction_bandit_intermediate(argdict):
    template = cifar_params_intermediate.template_func(argdict['param_args'])
    interpreted_template = scope.config_interpret_intermediate(template)
    return scope.cifar_prediction_bandit_evaluate(interpreted_template, argdict)


@scope.define
def config_interpret_intermediate(config):
    config = copy.deepcopy(config)
    config['layer_def'] = cifar_params_intermediate.config_interpretation(config['layer_def'])
    return config


@hyperopt.base.as_bandit(exceptions=bandit_exceptions)
def cifar_prediction_bandit_intermediate2(argdict):
    template = cifar_params_intermediate2.template_func(argdict['param_args'])
    interpreted_template = scope.config_interpret_intermediate2(template)
    return scope.cifar_prediction_bandit_evaluate(interpreted_template, argdict)


@scope.define
def config_interpret_intermediate2(config):
    config = copy.deepcopy(config)
    config['layer_def'] = cifar_params_intermediate2.config_interpretation(config['layer_def'])
    return config


@hyperopt.base.as_bandit(exceptions=bandit_exceptions)
def cifar_prediction_bandit_intermediate3(argdict):
    template = cifar_params_intermediate3.template_func(argdict['param_args'])
    interpreted_template = scope.config_interpret_intermediate3(template)
    return scope.cifar_prediction_bandit_evaluate2(interpreted_template, argdict)


@scope.define
def config_interpret_intermediate3(config):
    config = copy.deepcopy(config)
    config['layer_def'] = cifar_params_intermediate3.config_interpretation(config['layer_def'])
    return config


@hyperopt.base.as_bandit(exceptions=bandit_exceptions)
def cifar_prediction_bandit_new(argdict):
    template = cifar_params_new.template_func(argdict['param_args'])
    interpreted_template = scope.config_interpret(template)
    return scope.cifar_prediction_bandit_evaluate(interpreted_template, argdict)


@scope.define
def config_interpret(template):
    return cifar_params_new.config_interpretation(template)


@scope.define
def cifar_prediction_bandit_evaluate(config, kwargs, features=None):
    _, layer_fname = tempfile.mkstemp()
    odict_to_config(config['layer_def'], savepath=layer_fname)
    _, layer_param_fname = tempfile.mkstemp()
    odict_to_config(config['learning_params'], savepath=layer_param_fname)

    exp_id = kwargs['experiment_id']
    fs_name = 'cifar_prediction'
    config_str = json.dumps(config)
    config_id = hashlib.sha1(config_str).hexdigest()
    exp_str = json.dumps(collections.OrderedDict([("experiment_id", exp_id),
                          ("config", config),
                          ("config_id", config_id)]))

    op = ConvNet.get_options_parser()
    oppdict = [('--save-db', '1'),
               ('--save-recent-filters', '0'),
               ('--crop', '4'),
               ('--train-range', '0-4'),
               ('--test-range', '5'),
               ('--layer-def', layer_fname),
               ('--conserve-mem', '1'),
               ('--layer-params', layer_param_fname),
               ('--data-provider', 'general-cropped'),
               ('--dp-params', '{"preproc": {"normalize": false, "dtype": "float32", "mask": null, "crop": null, "resize_to": [32, 32], "mode": "RGB"}, "batch_size": 10000, "meta_attribute": "category", "dataset_name":["dldata.stimulus_sets.cifar10", "Cifar10"]}'),
               ('--test-freq', '50'),
               ('--saving-freq', '7'),
               ('--epochs', '100'),
               ('--img-size', '32'),
               ('--experiment-data', exp_str),
               ('--checkpoint-db-name', 'cifar_prediction'),
               ("--checkpoint-fs-name", fs_name)]
    gpu_num = os.environ.get('BANDIT_GPU', None)
    if gpu_num is not None:
        oppdict.append(('--gpu', gpu_num))

    op, load_dic = IGPUModel.parse_options(op, input_opts=collections.OrderedDict(oppdict), ignore_argv=True)
    nr.seed(0)
    model = ConvNet(op, load_dic)
    try:
        model.start()
    except SystemExit, e:
        if not e.code == 0:
            raise e

    cpt = IGPUModel.load_checkpoint_from_db({"experiment_data.experiment_id":exp_id, "experiment_data.config_id": config_id}, checkpoint_fs_host='localhost', checkpoint_fs_port=27017, checkpoint_db_name='cifar_prediction', checkpoint_fs_name=fs_name, only_rec=True)
    rec = cpt['rec']
    rec['kwargs'] = kwargs
    rec['loss'] = rec['test_outputs'][0]['logprob'][0]
    rec['status'] = 'ok'

    return rec


def reduce_learning_rates(config, factor):
    for l in config:
        for k in l:
            if 'eps' in k:
                config[l][k] *= factor


@scope.define
def cifar_prediction_bandit_evaluate2(config, kwargs, features=None):
    _, layer_fname = tempfile.mkstemp()
    odict_to_config(config['layer_def'], savepath=layer_fname)
    _, layer_param_fname = tempfile.mkstemp()
    odict_to_config(config['learning_params'], savepath=layer_param_fname)

    exp_id = kwargs['experiment_id']
    fs_name = 'cifar_prediction'
    config_str = json.dumps(config)
    config_id = hashlib.sha1(config_str).hexdigest()
    exp_str = json.dumps(collections.OrderedDict([("experiment_id", exp_id),
                          ("config", config),
                          ("config_id", config_id)]))

    op = ConvNet.get_options_parser()
    oppdict = [('--save-db', '1'),
               ('--save-recent-filters', '0'),
               ('--crop', '4'),
               ('--train-range', '0-4'),
               ('--test-range', '5'),
               ('--layer-def', layer_fname),
               ('--conserve-mem', '1'),
               ('--layer-params', layer_param_fname),
               ('--data-provider', 'general-cropped'),
               ('--dp-params', '{"preproc": {"normalize": false, "dtype": "float32", "mask": null, "crop": null, "resize_to": [32, 32], "mode": "RGB"}, "batch_size": 10000, "meta_attribute": "category", "dataset_name":["dldata.stimulus_sets.cifar10", "Cifar10"]}'),
               ('--test-freq', kwargs.get('test_freq', 50)),
               ('--saving-freq', '7'),
               ('--epochs', kwargs.get('epochs_round0', 75)),
               ('--img-size', '32'),
               ('--experiment-data', exp_str),
               ('--checkpoint-db-name', 'cifar_prediction'),
               ("--checkpoint-fs-name", fs_name)]
    gpu_num = os.environ.get('BANDIT_GPU', None)
    if gpu_num is not None:
        oppdict.append(('--gpu', gpu_num))

    op, load_dic = IGPUModel.parse_options(op, input_opts=collections.OrderedDict(oppdict), ignore_argv=True)
    nr.seed(0)
    model = ConvNet(op, load_dic)
    try:
        model.start()
    except SystemExit, e:
        if not e.code == 0:
            raise e
            
    model.scale_learningRate(0.1)
    model.num_epochs = kwargs.get('epochs_round1', 100)
    try:
        model.start()
    except SystemExit, e:
        if not e.code == 0:
            raise e
            

    cpt = IGPUModel.load_checkpoint_from_db({"experiment_data.experiment_id":exp_id, "experiment_data.config_id": config_id}, checkpoint_fs_host='localhost', checkpoint_fs_port=27017, checkpoint_db_name='cifar_prediction', checkpoint_fs_name=fs_name, only_rec=True)
    rec = cpt['rec']
    rec['kwargs'] = kwargs
    rec['loss'] = rec['test_outputs'][0]['logprob'][0]
    rec['status'] = 'ok'

    return rec



#cmd_tmpl  = """python convnet.py --crop=%d --test-range=0-4 --train-range=5 --layer-def=%s --layer-params=%s --data-provider=general-cropped  --dp-params='{"preproc": {"normalize": false, "dtype": "float32", "mask": null, "crop": null, "resize_to": [32, 32], "mode": "RGB"}, "batch_size": 10000, "meta_attribute": "category", "dataset_name":["dldata.stimulus_sets.cifar10", "Cifar10"]}' --test-freq=%d --epochs=%d --save-db=1 --img-size=32 --experiment-data='{"experiment_id":"%s", "config":%s, "config_id":"%s"}' --checkpoint-fs-name=%s"""  % (crop, layer_fname, layer_param_fname, tf, epochs, exp_id, config_str, config_id, fs_name)
    #retcode = subprocess.call(cmd_tmpl, shell=True)
