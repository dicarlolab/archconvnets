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

from . import imgnet_params_expanded
from . import imgnet_params_intermediaten1_filtersz_comb
from . import imgnet_params_intermediaten1_filtersz_avgpool
from . import imgnet_params_intermediaten1_filtersz_poolnormsz
from . import imgnet_params_intermediaten1_filtersz_normparams
from . import imgnet_params_intermediaten1_filtersz
from . import imgnet_params_intermediaten1
from . import imgnet_params_intermediate0
from . import imgnet_params_intermediate1
from . import imgnet_params_intermediate2
from . import imgnet_params_intermediate3
from .hyperopt_helpers import suggest_multiple_from_name

def imgnet_tpe_experiment_expanded(experiment_id):
    dbname = 'imgnet_predictions_tpe_experiment_expanded'
    host = 'localhost'
    port = 6666
    bandit = 'imgnet_prediction_bandit_expanded'
    bandit_kwargdict = {'param_args': {}, 'experiment_id': experiment_id,
                        'epochs_round0': 1, 'epochs_round1': 1}
    exp = imgnet_tpe_experiment(dbname, host, port, bandit, bandit_kwargdict,
                              num=1,
                              gamma=0.25,
                              n_startup_jobs=200)
    return exp

def imgnet_random_experiment_expanded(experiment_id):
    dbname = 'imgnet_predictions_random_experiment_expanded'
    host = 'localhost'
    port = 22334
    bandit = 'imgnet_prediction_bandit_expanded'
    bandit_kwargdict = {'param_args': {}, 'experiment_id': experiment_id}
    exp = imgnet_random_experiment(dbname, host, port, bandit, bandit_kwargdict)
    return exp

def imgnet_random_experiment_intermediaten1_filtersz_comb(experiment_id):
    dbname = 'imgnet_predictions_random_experiment_comb'
    host = 'localhost'
    port = 6667
    bandit = 'imgnet_prediction_bandit_intermediaten1_filtersz_comb'
    bandit_kwargdict = {'param_args': {}, 'experiment_id': experiment_id}
    exp = imgnet_random_experiment(dbname, host, port, bandit, bandit_kwargdict)
    return exp

def imgnet_random_experiment_intermediaten1_filtersz_avgpool(experiment_id):
    dbname = 'imgnet_predictions_random_experiment_intermediaten1_avgpool'
    host = 'localhost'
    port = 6667
    bandit = 'imgnet_prediction_bandit_intermediaten1_filtersz_avgpool'
    bandit_kwargdict = {'param_args': {}, 'experiment_id': experiment_id}
    exp = imgnet_random_experiment(dbname, host, port, bandit, bandit_kwargdict)
    return exp

def imgnet_random_experiment_intermediaten1_filtersz_normparams(experiment_id):
    dbname = 'imgnet_predictions_random_experiment_intermediaten1_normparams'
    host = 'localhost'
    port = 6667
    bandit = 'imgnet_prediction_bandit_intermediaten1_filtersz_normparams'
    bandit_kwargdict = {'param_args': {}, 'experiment_id': experiment_id}
    exp = imgnet_random_experiment(dbname, host, port, bandit, bandit_kwargdict)
    return exp

def imgnet_random_experiment_intermediaten1_filtersz_poolnormsz(experiment_id):
    dbname = 'imgnet_predictions_random_experiment_intermediaten1_poolnormsz'
    host = 'localhost'
    port = 6667
    bandit = 'imgnet_prediction_bandit_intermediaten1_filtersz_poolnormsz'
    bandit_kwargdict = {'param_args': {}, 'experiment_id': experiment_id}
    exp = imgnet_random_experiment(dbname, host, port, bandit, bandit_kwargdict)
    return exp

def imgnet_random_experiment_intermediaten1_filtersz(experiment_id):
    dbname = 'imgnet_predictions_random_experiment_intermediaten1_filtersz'
    host = 'localhost'
    port = 6667
    bandit = 'imgnet_prediction_bandit_intermediaten1_filtersz'
    bandit_kwargdict = {'param_args': {}, 'experiment_id': experiment_id}
    exp = imgnet_random_experiment(dbname, host, port, bandit, bandit_kwargdict)
    return exp

def imgnet_random_experiment_intermediaten1(experiment_id):
    dbname = 'imgnet_predictions_random_experiment_intermediaten1'
    host = 'localhost'
    port = 6667
    bandit = 'imgnet_prediction_bandit_intermediaten1'
    bandit_kwargdict = {'param_args': {}, 'experiment_id': experiment_id}
    exp = imgnet_random_experiment(dbname, host, port, bandit, bandit_kwargdict)
    return exp

def imgnet_random_experiment_intermediate0(experiment_id):
    dbname = 'imgnet_predictions_random_experiment_intermediate0'
    host = 'localhost'
    port = 6667
    bandit = 'imgnet_prediction_bandit_intermediate0'
    bandit_kwargdict = {'param_args': {}, 'experiment_id': experiment_id}
    exp = imgnet_random_experiment(dbname, host, port, bandit, bandit_kwargdict)
    return exp

def imgnet_random_experiment_intermediate1(experiment_id):
    dbname = 'imgnet_predictions_random_experiment_intermediate1'
    host = 'localhost'
    port = 6667
    bandit = 'imgnet_prediction_bandit_intermediate1'
    bandit_kwargdict = {'param_args': {}, 'experiment_id': experiment_id}
    exp = imgnet_random_experiment(dbname, host, port, bandit, bandit_kwargdict)
    return exp

def imgnet_random_experiment_intermediate2(experiment_id):
    dbname = 'imgnet_predictions_random_experiment_intermediate2'
    host = 'localhost'
    port = 6667
    bandit = 'imgnet_prediction_bandit_intermediate2'
    bandit_kwargdict = {'param_args': {}, 'experiment_id': experiment_id}
    exp = imgnet_random_experiment(dbname, host, port, bandit, bandit_kwargdict)
    return exp

def imgnet_random_experiment_intermediate3(experiment_id):
    dbname = 'imgnet_predictions_random_experiment_intermediate3'
    host = 'localhost'
    port = 6667
    bandit = 'imgnet_prediction_bandit_intermediate3'
    bandit_kwargdict = {'param_args': {}, 'experiment_id': experiment_id}
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


def imgnet_tpe_experiment(dbname, host, port, bandit, bandit_kwargdict,
                           num,
                           gamma,
                           n_startup_jobs):
    bandit_algo_names = ['hyperopt.TreeParzenEstimator'] * num
    bandit_names = ['archconvnets.hyperopt.imgnet_prediction.%s' % bandit] * num
    #ek = hashlib.sha1(cPickle.dumps(bandit_kwargdict)).hexdigest()
    exp_keys = ['imgnet_prediction_tpe_%s_%s_%i' % (bandit, bandit_kwargdict['experiment_id'], i) for i in range(num)]
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
def imgnet_prediction_bandit_expanded(argdict):
    template = imgnet_params_expanded.template_func(argdict['param_args'])
    interpreted_template = scope.config_interpret_expanded(template)
    return scope.imgnet_prediction_bandit_evaluate2(interpreted_template, argdict)

@hyperopt.base.as_bandit(exceptions=bandit_exceptions)
def imgnet_prediction_bandit_intermediaten1_filtersz_comb(argdict):
    template = imgnet_params_intermediaten1_filtersz_comb.template_func(argdict['param_args'])
    interpreted_template = scope.config_interpret_intermediaten1_filtersz_comb(template)
    return scope.imgnet_prediction_bandit_evaluate2(interpreted_template, argdict)

@hyperopt.base.as_bandit(exceptions=bandit_exceptions)
def imgnet_prediction_bandit_intermediaten1_filtersz_avgpool(argdict):
    template = imgnet_params_intermediaten1_filtersz_avgpool.template_func(argdict['param_args'])
    interpreted_template = scope.config_interpret_intermediaten1_filtersz_avgpool(template)
    return scope.imgnet_prediction_bandit_evaluate2(interpreted_template, argdict)

@hyperopt.base.as_bandit(exceptions=bandit_exceptions)
def imgnet_prediction_bandit_intermediaten1_filtersz_normparams(argdict):
    template = imgnet_params_intermediaten1_filtersz_normparams.template_func(argdict['param_args'])
    interpreted_template = scope.config_interpret_intermediaten1_filtersz_normparams(template)
    return scope.imgnet_prediction_bandit_evaluate2(interpreted_template, argdict)

@hyperopt.base.as_bandit(exceptions=bandit_exceptions)
def imgnet_prediction_bandit_intermediaten1_filtersz_poolnormsz(argdict):
    template = imgnet_params_intermediaten1_filtersz_poolnormsz.template_func(argdict['param_args'])
    interpreted_template = scope.config_interpret_intermediaten1_filtersz_poolnormsz(template)
    return scope.imgnet_prediction_bandit_evaluate2(interpreted_template, argdict)

@hyperopt.base.as_bandit(exceptions=bandit_exceptions)
def imgnet_prediction_bandit_intermediaten1_filtersz(argdict):
    template = imgnet_params_intermediaten1_filtersz.template_func(argdict['param_args'])
    interpreted_template = scope.config_interpret_intermediaten1_filtersz(template)
    return scope.imgnet_prediction_bandit_evaluate2(interpreted_template, argdict)

@hyperopt.base.as_bandit(exceptions=bandit_exceptions)
def imgnet_prediction_bandit_intermediaten1(argdict):
    template = imgnet_params_intermediaten1.template_func(argdict['param_args'])
    interpreted_template = scope.config_interpret_intermediaten1(template)
    return scope.imgnet_prediction_bandit_evaluate2(interpreted_template, argdict)

@hyperopt.base.as_bandit(exceptions=bandit_exceptions)
def imgnet_prediction_bandit_intermediate0(argdict):
    template = imgnet_params_intermediate0.template_func(argdict['param_args'])
    interpreted_template = scope.config_interpret_intermediate0(template)
    return scope.imgnet_prediction_bandit_evaluate2(interpreted_template, argdict)

@hyperopt.base.as_bandit(exceptions=bandit_exceptions)
def imgnet_prediction_bandit_intermediate1(argdict):
    template = imgnet_params_intermediate1.template_func(argdict['param_args'])
    interpreted_template = scope.config_interpret_intermediate1(template)
    return scope.imgnet_prediction_bandit_evaluate2(interpreted_template, argdict)

@hyperopt.base.as_bandit(exceptions=bandit_exceptions)
def imgnet_prediction_bandit_intermediate2(argdict):
    template = imgnet_params_intermediate2.template_func(argdict['param_args'])
    interpreted_template = scope.config_interpret_intermediate2(template)
    return scope.imgnet_prediction_bandit_evaluate2(interpreted_template, argdict)

@hyperopt.base.as_bandit(exceptions=bandit_exceptions)
def imgnet_prediction_bandit_intermediate3(argdict):
    template = imgnet_params_intermediate3.template_func(argdict['param_args'])
    interpreted_template = scope.config_interpret_intermediate3(template)
    return scope.imgnet_prediction_bandit_evaluate2(interpreted_template, argdict)

@scope.define
def config_interpret_expanded(config):
    config = copy.deepcopy(config)
    config['layer_def'] = imgnet_params_expanded.config_interpretation(config['layer_def'])
    return config

@scope.define
def config_interpret_intermediaten1_filtersz_comb(config):
    config = copy.deepcopy(config)
    config['layer_def'] = imgnet_params_intermediaten1_filtersz_comb.config_interpretation(config['layer_def'])
    return config

@scope.define
def config_interpret_intermediaten1_filtersz_avgpool(config):
    config = copy.deepcopy(config)
    config['layer_def'] = imgnet_params_intermediaten1_filtersz_avgpool.config_interpretation(config['layer_def'])
    return config

@scope.define
def config_interpret_intermediaten1_filtersz_poolnormsz(config):
    config = copy.deepcopy(config)
    config['layer_def'] = imgnet_params_intermediaten1_filtersz_poolnormsz.config_interpretation(config['layer_def'])
    return config

@scope.define
def config_interpret_intermediaten1_filtersz_normparams(config):
    config = copy.deepcopy(config)
    config['layer_def'] = imgnet_params_intermediaten1_filtersz_normparams.config_interpretation(config['layer_def'])
    return config

@scope.define
def config_interpret_intermediaten1_filtersz(config):
    config = copy.deepcopy(config)
    config['layer_def'] = imgnet_params_intermediaten1_filtersz.config_interpretation(config['layer_def'])
    return config

@scope.define
def config_interpret_intermediaten1(config):
    config = copy.deepcopy(config)
    config['layer_def'] = imgnet_params_intermediaten1.config_interpretation(config['layer_def'])
    return config

@scope.define
def config_interpret_intermediate3(config):
    config = copy.deepcopy(config)
    config['layer_def'] = imgnet_params_intermediate3.config_interpretation(config['layer_def'])
    return config

@scope.define
def config_interpret_intermediate2(config):
    config = copy.deepcopy(config)
    config['layer_def'] = imgnet_params_intermediate2.config_interpretation(config['layer_def'])
    return config

@scope.define
def config_interpret_intermediate1(config):
    config = copy.deepcopy(config)
    config['layer_def'] = imgnet_params_intermediate1.config_interpretation(config['layer_def'])
    return config

@scope.define
def config_interpret_intermediate0(config):
    config = copy.deepcopy(config)
    config['layer_def'] = imgnet_params_intermediate0.config_interpretation(config['layer_def'])
    return config

def reduce_learning_rates(config, factor):
    for l in config:
        for k in l:
            if 'eps' in k:
                config[l][k] *= factor


@scope.define
def imgnet_prediction_bandit_evaluate2(config, kwargs, features=None):
    _, layer_fname = tempfile.mkstemp()
    odict_to_config(config['layer_def'], savepath=layer_fname)
    _, layer_param_fname = tempfile.mkstemp()
    odict_to_config(config['learning_params'], savepath=layer_param_fname)

    exp_id = kwargs['experiment_id']
    fs_name = 'imgnet_prediction'
    config_str = json.dumps(config)
    config_id = hashlib.sha1(config_str).hexdigest()
    exp_str = json.dumps(collections.OrderedDict([("experiment_id", exp_id),
                          ("config", config),
                          ("config_id", config_id)]))

    op = ConvNet.get_options_parser()
    oppdict = [('--save-db', '1'),
               ('--save-filters','1'),
               ('--save-recent-filters', '1'),('--save-recent','1'),
               ('--crop-border', '9'),
               ('--train-range', '0-1250'),#'0-4'),
               ('--test-range', '4351-4550'),#'5'),
               ('--layer-def', layer_fname),
               ('--conserve-mem', '1'),
               ('--layer-params', layer_param_fname),
               ('--checkpoint-fs-port', '6666'),
               ('--data-provider', 'general-cropped'),
               ('--data-path', '/storage/imgnet_256batchsz_138px'),
               ('--dp-params', '{"perm_type": "random", "perm_seed": 0, "preproc": {"normalize": false, "dtype": "float32", "resize_to": [138, 138], "mode": "RGB", "crop": null, "mask": null}, "batch_size": 256, "meta_attribute": "synset", "dataset_name": ["imagenet.dldatasets", "ChallengeSynsets2013_offline"]}'),
               ('--test-freq', kwargs.get('test_freq', 100)),
               ('--saving-freq', '0'),
               ('--epochs', kwargs.get('epochs_round0', 5)),
               ('--img-size', '138'),
               ('--experiment-data', exp_str),
               ('--checkpoint-db-name', 'imgnet_prediction'),
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
    model.num_epochs = kwargs.get('epochs_round1', 7)
    try:
        model.start()
    except SystemExit, e:
        if not e.code == 0:
            raise e
            

    print exp_id
    print config_id
    cpt = IGPUModel.load_checkpoint_from_db({"experiment_data.experiment_id":exp_id, "experiment_data.config_id": config_id}, checkpoint_fs_host='localhost', checkpoint_fs_port=6666, checkpoint_db_name='imgnet_prediction', checkpoint_fs_name=fs_name, only_rec=True)
    rec = cpt['rec']
    rec['kwargs'] = kwargs
    rec['loss'] = rec['test_outputs'][0]['logprob'][0]
    rec['status'] = 'ok'

    return rec



#cmd_tmpl  = """python convnet.py --crop=%d --test-range=0-4 --train-range=5 --layer-def=%s --layer-params=%s --data-provider=general-cropped  --dp-params='{"preproc": {"normalize": false, "dtype": "float32", "mask": null, "crop": null, "resize_to": [32, 32], "mode": "RGB"}, "batch_size": 10000, "meta_attribute": "category", "dataset_name":["dldata.stimulus_sets.cifar10", "Cifar10"]}' --test-freq=%d --epochs=%d --save-db=1 --img-size=32 --experiment-data='{"experiment_id":"%s", "config":%s, "config_id":"%s"}' --checkpoint-fs-name=%s"""  % (crop, layer_fname, layer_param_fname, tf, epochs, exp_id, config_str, config_id, fs_name)
    #retcode = subprocess.call(cmd_tmpl, shell=True)
