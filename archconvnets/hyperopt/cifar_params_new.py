import copy
from collections import OrderedDict
import numpy as np

try:
    from hyperopt.pyll import scope
except ImportError:
    print 'Trying standalone pyll'
    from pyll import scope
from hyperopt.pyll_utils import hp_uniform, hp_loguniform, hp_quniform, hp_qloguniform
from hyperopt.pyll_utils import hp_normal, hp_lognormal, hp_qnormal, hp_qlognormal
from hyperopt.pyll_utils import hp_choice


conv_template = lambda i : [OrderedDict([('type', 'conv'),
                                    ('filters', scope.int(hp_qloguniform('conv_num_filters_%d' % i, np.log(16), np.log(96), q=16))),
                                    ('padding',  hp_choice('conv_padding_%d' % i, [1, 2])),
                                    ('stride', hp_choice('conv_stride_%d' % i, [1, 2])),
                                    ('filterSize', scope.int(hp_quniform('conv_filter_shape_%d' % i, 2, 8, 1))),
                                    ('neuron', 'relu'),
                                    ('initw', hp_choice('conv_initw_%d' % i, [1e-4, 1e-3, 1e-2])),
                                    ('partialsum', 1),
                                    ('sharedbiases', 1)]),  
                 OrderedDict([('epsw', hp_choice('conv_epsw_%d' % i, [1e-4, 1e-3, 1e-2])),
                               ('epsb', hp_choice('conv_epsb_%d' % i, [2e-4, 2e-3, 2e-2])),
                               ('momw', hp_uniform('conv_momw_%d' % i, .55, .95)),
                               ('momb', hp_uniform('conv_momb_%d' % i, .55, .95)),
                               ('wc', 0)])]

pool_template = lambda i : [OrderedDict([('type', 'pool'),
                                    ('pool', hp_choice('pool_type_%d' % i, 
                                                        ['max', 
                                                         'avg',
                                                         hp_uniform('pool_order_%d' % i, 0.4, 12)])),
                                    ('start', 0),
                                    ('sizeX', scope.int(hp_quniform('pool_sizex_%d' % i, 2, 5, 1))),
                                    ('stride', hp_choice('pool_stride_%d' % i, [1, 2])),
                                    ('outputsX', 0)]),
                None]


norm_template = lambda i: [OrderedDict([('type', 'cmrnorm'),
                              ('size', scope.int(hp_quniform('rnorm_size_%d' % i, 5, 12, 1)))]), 
                 OrderedDict([('scale', hp_choice('norm_scale_%d' % i, [1e-4, 1e-3, 1e-2])),
                              ('pow', hp_uniform('norm_pow_%d' % i, .55, .95))])]


local_template = lambda i : [OrderedDict([('type', 'local'),
                                     ('filters', scope.int(hp_qloguniform('local_num_filters_%d' % i, np.log(16), np.log(96), q=16))),
                                     ('padding', hp_choice('clocal_padding_%d' % i, [1, 2])),
                                     ('stride', hp_choice('local_stride_%d' % i, [1, 2])),
                                     ('filterSize', scope.int(hp_quniform('local_filter_shape_%d' % i, 2, 12, 1))),
                                     ('neuron', 'relu'),
                                     ('initw', hp_choice('local_initw_%d' % i, [1e-4, 1e-3, 1e-2, 1e-1]))]), 
                  OrderedDict([('epsw', hp_choice('local_epsw_%d' % i, [1e-4, 1e-3, 1e-2])),
                               ('epsb', hp_choice('local_epsb_%d' % i, [2e-4, 2e-3, 2e-2])),
                               ('momw', hp_uniform('local_momw_%d' % i, .55, .95)),
                               ('momb', hp_uniform('local_momb_%d' % i, .55, .95)),
                               ('wc', 0.004)])]

def channel_layers(N):
    return [hp_choice('channel_node_%d' % i, [conv_template(i), 
                                                pool_template(i),
                                                norm_template(i),
                                                local_template(i)]) 
                                            for i in range(N)]

def channel_layers_nolocal(N):
    return [hp_choice('channel_node_%d' % i, [conv_template(i), 
                                                pool_template(i),
                                                norm_template(i)]) 
                                            for i in range(N)]

final_layers = OrderedDict([('fc1', OrderedDict([('type', 'fcdropo'),
                                    ('outputs', scope.int(hp_qloguniform('fc1_num_outputs', np.log(64), np.log(256), q=64))),
                                    ('initw', hp_choice('fc1_initw', [1e-4, 1e-3, 1e-2])),
                                    ('neuron', 'relu'),
                                    ('rate', 0.5)])),
             ('fc10', OrderedDict([('type', 'fc'),
                                   ('outputs', '10'),
                                   ('inputs', 'fc1'),
                                   ('initw', hp_choice('fc10_initw', [1e-4, 1e-3, 1e-2]))])),
             ('probs', OrderedDict([('type', 'softmax'),
                                    ('inputs', 'fc10')])),
             ('logprob', OrderedDict([('type', 'cost.logreg'),
                                      ('inputs', 'labels,probs')]))])
                                      

final_layers_learning_params = OrderedDict([('fc1', OrderedDict([('epsw', hp_choice('fc1_epsw', [1e-4, 1e-3, 1e-2])),
                                           ('epsb', hp_choice('fc1_epsb', [2e-4, 2e-3, 2e-2])),
                                           ('momw', hp_uniform('fc1_momw', .55, .95)),
                                           ('momb', hp_uniform('fc1_momb', .55, .95)),
                                           ('wc', 0.004)])),
                                            ('fc10', OrderedDict([('epsw', hp_choice('fc10_epsw', [1e-4, 1e-3, 1e-2])),
                                           ('epsb', hp_choice('fc10_epsb', [2e-4, 2e-3, 2e-2])),
                                           ('momw', hp_uniform('fc10_momw', .55, .95)),
                                           ('momb', hp_uniform('fc10_momb', .55, .95)),
                                           ('wc', 0.01)])),
                                           ('logprob', OrderedDict([('coeff', '1')]))])
                                           
                                           
def template_func(args):
    return {'channel_layers': channel_layers_nolocal(args['num_layers']),
            'final_layers': final_layers, 
            'final_params': final_layers_learning_params}
            

def config_interpretation(config):
    layers = OrderedDict([('data', OrderedDict([('type', 'data'),
                                   ('dataidx', '0')])),
                          ('labels', OrderedDict([('type', 'data'),
                                    ('dataidx', '1')]))])
    params = OrderedDict([])
    
    n1 = 'channel_layer_0_%s' % config['channel_layers'][0][0]['type']
    layers[n1] = config['channel_layers'][0][0]
    layers[n1]['channels'] = 3
    layers[n1]['inputs'] = 'data'
    if config['channel_layers'][0][1] is not None:
        params[n1] = config['channel_layers'][0][1]
    
    i = 1
    for cn, cnl in config['channel_layers'][1:]:
        n1 = 'channel_layer_%d_%s' % (i, cn['type'])
        n0 = 'channel_layer_%d_%s' % (i-1, config['channel_layers'][i-1][0]['type'])
        layers[n1] = cn
        layers[n1]['inputs'] = n0
        if layers[n0]['type'] in ['local', 'conv']:
            layers[n1]['channels'] = layers[n0]['filters']
        else:
            layers[n1]['channels'] = layers[n0]['channels']
        
        if cnl is not None:
            params[n1] = cnl
        
        i += 1

    final_layers = copy.deepcopy(config['final_layers'])
    final_layers['fc1']['inputs'] = n1

    layers.update(final_layers)
    params.update(config['final_params'])

    newlayers = OrderedDict([])
    for (l_ind, l) in enumerate(layers):            
        if layers[l]['type'] == 'pool' and isinstance(layers[l]['pool'], int):
            order = layers[l]['pool']
            layers[l]['pool'] = 'avg'
            l00 = OrderedDict([('type', 'neuron'),
                              ('neuron', 'abs'),
                              ('inputs', layers[l]['inputs'])])
            l00name = l + '_prepre'
            l0 = OrderedDict([('type', 'neuron'),
                              ('neuron', 'power[%.4f]' % order),
                              ('inputs', l00name)])
            l0name = l + '_pre'
            layers[l]['inputs'] = l0name
            l10 = OrderedDict([('type', 'neuron'),
                               ('neuron', 'linear[1, 0.00001]'),
                               ('inputs', l)])
            l10name = l + '_postpre'
            l1 = OrderedDict([('type', 'neuron'),
                              ('neuron', 'power[%.4f]' % (1./order)),
                              ('inputs', l10name)
                              ])
            l1name = l + '_post'
            layers[layers.keys()[l_ind + 1]]['inputs'] = l1name
            newlayers[l00name] = l00
            newlayers[l0name] = l0
            newlayers[l] = layers[l]
            newlayers[l10name] = l10
            newlayers[l1name] = l1
        else:
            newlayers[l] = layers[l]

    return {'layer_def': newlayers,
            'learning_params': params}
