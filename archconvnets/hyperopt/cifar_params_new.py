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


conv_template = [OrderedDict([('type', 'conv'),
                                    ('filters', scope.int(hp_qloguniform('conv_num_filters_%d' % i, np.log(16), np.log(96), q=16))),
                                    ('padding',  hp_choice('conv_padding_%d' % i, [1, 2])),
                                    ('stride', hp_choice('conv_stride_%d' % i, [1, 2])),
                                    ('filtersize', scope.int(hp_quniform('conv_filter_shape_%d' % i, 2, 12, 1))),
                                    ('neuron', 'relu'),
                                    ('initw', hp_choice('conv_initw_%d' % i, [1e-4, 1e-3, 1e-2, 1e-1])),
                                    ('partialsum', 1),
                                    ('sharedbiases', 1)]),  
                 OrderedDict([('epsw', hp_choice('conv_epsw_%d' % i, [1e-3, 1e-2, 1e-1])),
                               ('epsb', hp_choice('conv_epsb_%d' % i, [2e-4, 2e-3, 2e-2])),
                               ('momw', hp_uniform('conv_momw_%d' % i, .55, .95)),
                               ('momb', hp_uniform('conv_momb_%d' % i, .55, .95)),
                               ('wc', 0)])]

pool_template = [OrderedDict([('type', 'pool'),
                                    ('pool', hp_choice('pool_type_%d' % i, 
                                                        ['max', 
                                                         'avg',
                                                         scope.int(hp_quniform('pool_order_%d' % i, 2, 12, q=2))])),
                                    ('start', 0),
                                    ('sizex', scope.int(hp_quniform('pool_sizex_%d' % i, 2, 5, 1))),
                                    ('stride', hp_choice('pool_stride_%d' % i, [1, 2])),
                                    ('outputsx', 0)]),
                None]


norm_template = [OrderedDict([('type', 'cmrnorm'),
                              ('size', scope.int(hp_quniform('rnorm_size_%d' % i, 5, 12, 1)))]), 
                 OrderedDict([('scale', hp_choice('norm_scale_%d' % i, [1e-4, 1e-3, 1e-2])),
                              ('pow', hp_uniform('norm_pow_%d' % i, .55, .95))])]


local_template = [OrderedDict([('type', 'local'),
                                     ('filters', scope.int(hp_qloguniform('local_num_filters_%d' % i, np.log(16), np.log(96), q=16))),
                                     ('padding', hp_choice('clocal_padding_%d' % i, [1, 2])),
                                     ('stride', hp_choice('local_stride_%d' % i, [1, 2])),
                                     ('filtersize', scope.int(hp_quniform('local_filter_shape_%d' % i, 2, 12, 1))),
                                     ('neuron', 'relu'),
                                     ('initw', hp_choice('local_initw_%d' % i, [1e-4, 1e-3, 1e-2, 1e-1]))]), 
                  OrderedDict([('epsw', hp_choice('local_epsw_%d' % i, [1e-3, 1e-2, 1e-1])),
                               ('epsb', hp_choice('local_epsb_%d' % i, [2e-4, 2e-3, 2e-2])),
                               ('momw', hp_uniform('local_momw_%d' % i, .55, .95)),
                               ('momb', hp_uniform('local_momb_%d' % i, .55, .95)),
                               ('wc', 0.004)])]

def channel_layers(N):
    return [hp_choice('channel_node_%d' % i, [conv_template , 
                                                pool_template,
                                                norm_template,
                                                local_template]) 
                                            for i in range(N)]

def channel_layers_nolocal(N):
    return [hp_choice('channel_node_%d' % i, [conv_template , 
                                                pool_template,
                                                norm_template]) 
                                            for i in range(N)]

final_layers = OrderedDict([('fc1', OrderedDict([('type', 'fcdropo'),
                                    ('outputs', scope.int(hp_qloguniform('fc1_num_outputs', np.log(64), np.log(256), q=64))),
                                    ('inputs', 'channel_node_9'),
                                    ('initw', hp_choice('fc1_initw', [1e-3, 1e-2, 1e-1])),
                                    ('neuron', 'relu'),
                                    ('rate', 0.5)])),
             ('fc10', OrderedDict([('type', 'fc'),
                                   ('outputs', '10'),
                                   ('inputs', 'fc1'),
                                   ('initw', hp_choice('fc10_initw', [1e-3, 1e-2, 1e-1]))])),
             ('probs', OrderedDict([('type', 'softmax'),
                                    ('inputs', 'fc10')])),
             ('logprob', OrderedDict([('type', 'cost.logreg'),
                                      ('inputs', 'labels,probs')]))])
                                      

final_layers_learning_params = OrderedDict([('fc128', OrderedDict([('epsw', hp_choice('fc1_epsw_%d' % i, [1e-3, 1e-2, 1e-1])),
                                           ('epsb', hp_choice('fc1_epsb_%d' % i, [2e-4, 2e-3, 2e-2])),
                                           ('momw', hp_uniform('fc1_momw_%d' % i, .55, .95)),
                                           ('momb', hp_uniform('fc1_momb_%d' % i, .55, .95)),
                                           ('wc', 0.004)])),
                                            ('fc10', OrderedDict([('epsw', hp_choice('fc10_epsw_%d' % i, [1e-3, 1e-2, 1e-1])),
                                           ('epsb', hp_choice('fc10_epsb_%d' % i, [2e-4, 2e-3, 2e-2])),
                                           ('momw', hp_uniform('fc10_momw_%d' % i, .55, .95)),
                                           ('momb', hp_uniform('fc10_momb_%d' % i, .55, .95)),
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
    layers[n1]['input'] = 'data'
    params[n1] = config['channel_layers'][0][1]
    
    i = 1
    for cn, cnl in config['channel_layers'][1:]:
        n1 = 'channel_layer_%d_%s' % (i, cn['type'])
        n0 = 'channel_layer_%d_%s' % (i-1, config['channel_layers'][i-1][0]['type'])
        layers[n1] = cn
        layers[n1]['input'] = n0
        if layers[n0]['type'] in ['local', 'conv']:
            layers[n1]['channels'] = layers[n0]['filters']
        else:
            layers[n1]['channels'] = layers[n0]['channels']
        
        if cnl is not None:
            params[n1] = cnl
            
    
        
        i += 1
        
    layers.update(config['final_layers'])
    params.update(config['final_params'])
    
    return layers, params
    
                  
    
                            


                          