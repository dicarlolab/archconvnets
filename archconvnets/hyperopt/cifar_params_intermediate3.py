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


num_filters1 = scope.int(hp_qloguniform('num_filters1',np.log(16), np.log(96), q=16))
filter1_size = scope.int(hp_quniform('filter1_shape', 2, 12, 1))

num_filters2 = scope.int(hp_qloguniform('num_filters2',np.log(16), np.log(96), q=16))
filter2_size = scope.int(hp_quniform('filter2_shape', 2, 12, 1))

num_filters3 = scope.int(hp_qloguniform('num_filters3',np.log(16), np.log(96), q=16))
filter3_size = scope.int(hp_quniform('filter3_shape', 2, 9, 1))

num_filters4 = scope.int(hp_qloguniform('num_filters4',np.log(16), np.log(64), q=16))
filter4_size = scope.int(hp_quniform('filter4_shape', 2, 9, 1))

pool1_sizex = scope.int(hp_quniform('pool1_sizex', 2, 5, 1))
pool1_type = hp_choice('pool1_type', ['max', 'avg', hp_uniform('pool_order_1', 1, 12)])

pool2_sizex = scope.int(hp_quniform('pool2_sizex', 2, 5, 1))
pool2_type = hp_choice('pool2_type', ['max', 'avg', hp_uniform('pool_order_2', 1, 4)])

pool3_sizex = scope.int(hp_quniform('pool3_sizex', 2, 5, 1))
pool3_type = hp_choice('pool3_type', ['max', 'avg', hp_uniform('pool_order_3', 1, 4)])

rnorm1_size = scope.int(hp_quniform('rnorm1_size', 5, 12, 1))
rnorm2_size = scope.int(hp_quniform('rnorm2_size', 5, 12, 1))
rnorm3_size = scope.int(hp_quniform('rnorm3_size', 5, 12, 1))


layer_def_template = OrderedDict([('data', OrderedDict([('type', 'data'),
                                   ('dataidx', 0)])),
             ('labels', OrderedDict([('type', 'data'),
                                     ('dataidx', 1)])),
             ('conv1', OrderedDict([('type', 'conv'),
                                    ('inputs', 'data'),
                                    ('channels', 3),
                                    ('filters', num_filters1),
                                    ('padding', 2),
                                    ('stride', 1),
                                    ('filtersize', filter1_size),
                                    ('neuron', 'relu'),
                                    ('initw', hp_choice('initw_conv1', [1e-3, 5e-4, 1e-4])),
                                    ('partialsum', 1),
                                    ('sharedbiases', 1)])),
             ('pool1', hp_choice('pool1', [OrderedDict([('inputs', 'conv1'),
                                                        ('type', None)]), 
                                           OrderedDict([('type', 'pool'),
                                                        ('pool', pool1_type),
                                                        ('inputs', 'conv1'),
                                                        ('start', '0'),
                                                        ('sizex', pool1_sizex),
                                                        ('stride', 2),
                                                        ('outputsx', 0),
                                                        ('channels', num_filters1)])])),
             ('rnorm1', hp_choice('rn1', [OrderedDict([('inputs', 'pool1'),
                                                       ('type', None)]),
                                          OrderedDict([('type', 'cmrnorm'),
                                                       ('inputs', 'pool1'),
                                                       ('channels', num_filters1),
                                                       ('size', rnorm1_size)])])),
             ('conv2', OrderedDict([('type', 'conv'),
                                    ('inputs', 'rnorm1'),
                                    ('filters', num_filters2),
                                    ('padding', 2),
                                    ('stride', 1),
                                    ('filtersize', filter2_size),
                                    ('channels', num_filters1),
                                    ('neuron', 'relu'),
                                    ('initw', hp_choice('initw_conv2', [1e-1, 5e-2, 1e-2, 5e-3])),
                                    ('partialsum', 1),
                                    ('sharedbiases', 1)])),
             ('rnorm2', hp_choice('rn2', [OrderedDict([('inputs', 'conv2'),
                                                       ('type', None)]), 
                                          OrderedDict([('type', 'cmrnorm'),
                                                       ('inputs', 'conv2'),
                                                       ('channels', num_filters2),
                                                       ('size', rnorm2_size)])])),
             ('pool2', hp_choice('pool2', [OrderedDict([('inputs', 'rnorm2'),
                                                       ('type', None)]), 
                                           OrderedDict([('type', 'pool'),
                                                        ('pool', pool2_type),
                                                        ('inputs', 'rnorm2'),
                                                        ('start', 0),
                                                        ('sizex', pool2_sizex),
                                                        ('stride', 2),
                                                        ('outputsx', 0),
                                                        ('channels', num_filters2)])])),
             ('conv3', hp_choice('conv3', [OrderedDict([('type', 'conv'),
                                             ('inputs', 'pool2'),
                                             ('filters', num_filters3),
                                             ('padding', 1),
                                             ('stride', 1),
                                             ('filtersize', filter3_size),
                                             ('channels', num_filters2),
                                             ('neuron', 'relu'),
                                             ('initw', hp_choice('initw_conv3', [1e-1, 5e-2, 1e-2, 5e-3])),
                                             ('partialsum', 1),
                                             ('sharedbiases', 1)]),
                                           OrderedDict([('type', 'local'),
                                             ('inputs', 'pool2'),
                                             ('filters', num_filters3),
                                             ('padding', 1),
                                             ('stride', 1),
                                             ('filtersize', filter3_size),
                                             ('channels', num_filters2),
                                             ('neuron', 'relu'),
                                             ('initw', 0.04)])])),
             ('pool3', hp_choice('pool3', [OrderedDict([('inputs', 'conv3'),
                                                        ('type', None)]), 
                                           OrderedDict([('type', 'pool'),
                                                        ('pool', pool3_type),
                                                        ('inputs', 'conv3'),
                                                        ('start', '0'),
                                                        ('sizex', pool3_sizex),
                                                        ('stride', 1),
                                                        ('outputsx', 0),
                                                        ('channels', num_filters3)])])),
             ('rnorm3', hp_choice('rn3', [OrderedDict([('inputs', 'pool3'),
                                                       ('type', None)]),
                                          OrderedDict([('type', 'cmrnorm'),
                                                       ('inputs', 'pool3'),
                                                       ('channels', num_filters3),
                                                       ('size', rnorm3_size)])])),                                             
             ('conv4', hp_choice('conv4', [OrderedDict([('type', 'conv'),
                                                 ('inputs', 'conv3'),
                                                 ('filters', num_filters4),
                                                 ('padding', 1),
                                                 ('stride', 1),
                                                 ('filtersize', filter4_size),
                                                 ('channels', num_filters3),
                                                 ('neuron', 'relu'),
                                                 ('initw', hp_choice('initw_conv4', [1e-1, 5e-2, 1e-2, 5e-3])),
                                                 ('partialsum', 1),
                                                 ('sharedbiases', 1)]),
                                           OrderedDict([('type', 'local'),
                                                 ('inputs', 'conv3'),
                                                 ('filters', num_filters4),
                                                 ('padding', 1),
                                                 ('stride', 1),
                                                 ('filtersize', filter4_size),
                                                 ('channels', num_filters3),
                                                 ('neuron', 'relu'),
                                                 ('initw', 0.04)])])),
             ('fc128', hp_choice('fc', [OrderedDict([('type', 'fcdropo'),
                                                     ('outputs', 128),
                                                     ('inputs', 'conv4'),
                                                     ('initw', hp_choice('initw_fc128', [1e-1, 5e-2, 1e-2, 5e-3])),
                                                     ('neuron', 'relu'),
                                                     ('rate', 0.5)]),
                                        OrderedDict([('inputs', 'conv4'),
                                                     ('type', None)])])),
             ('fc10', OrderedDict([('type', 'fc'),
                                   ('outputs', 10),
                                   ('inputs', 'fc128'),
                                   ('initw', hp_choice('initw_fc10', [1e-1, 5e-2, 1e-2, 5e-3]))])),
             ('probs', OrderedDict([('type', 'softmax'),
                                    ('inputs', 'fc10')])),
             ('logprob', OrderedDict([('type', 'cost.logreg'),
                                      ('inputs', 'labels,probs')]))])


learning_params_template = OrderedDict([('conv1', OrderedDict([('epsw', hp_choice('epsw_conv1', [1e-2, 3e-3, 1e-3, 5e-4, 1e-4])),
                                                               ('epsb', hp_choice('epsb_conv1', [2e-2, 6e-3, 2e-3, 5e-4, 2e-4])),
                                                               ('momw', 0.9),
                                                               ('momb', 0.9),
                                                               ('wc', 0.000)])),
                                        ('rnorm1', OrderedDict([('scale', hp_choice('rnorm1_scale', [1e-2, 1e-3, 1e-4])),
                                                                ('pow', hp_uniform('rnorm1_pow', 0.6, 0.95))])),
                                        ('conv2', OrderedDict([('epsw', hp_choice('epsw_conv2', [1e-2, 3e-3, 1e-3, 5e-4, 1e-4])),
                                                               ('epsb', hp_choice('epsb_conv2', [2e-2, 6e-3, 2e-3, 5e-4, 2e-4])),
                                                               ('momw', 0.9),
                                                               ('momb', 0.9),
                                                               ('wc', 0.000)])),
                                        ('rnorm2', OrderedDict([('scale', 0.001),
                                                                ('pow', hp_uniform('rnorm2_pow', 0.6, 0.95))])),
                                        ('conv3', OrderedDict([('epsw', hp_choice('epsw_conv3', [1e-2, 3e-3, 1e-3, 5e-4, 1e-4])),
                                                               ('epsb', hp_choice('epsb_conv3', [2e-2, 6e-3, 2e-3, 5e-4, 2e-4])),
                                                               ('momw', 0.9),
                                                               ('momb', 0.9),
                                                               ('wc', 0.004)])),
                                        ('rnorm3', OrderedDict([('scale', 0.001),
                                                                ('pow', hp_uniform('rnorm3_pow', 0.6, 0.95))])),                                                                
                                        ('conv4', OrderedDict([('epsw', hp_choice('epsw_conv4', [1e-2, 3e-3, 1e-3, 5e-4, 1e-4])),
                                                               ('epsb', hp_choice('epsb_conv4', [2e-2, 6e-3, 2e-3, 5e-4, 2e-4])),
                                                               ('momw', 0.9),
                                                               ('momb', 0.9),
                                                               ('wc', 0.004)])),
                                        ('fc128', OrderedDict([('epsw', hp_choice('epsw_fc128', [1e-2, 3e-3, 1e-3, 5e-4, 1e-4])),
                                                              ('epsb', hp_choice('epsb_fc128', [2e-2, 6e-3, 2e-3, 5e-4, 2e-4])),
                                                              ('momw', 0.9),
                                                              ('momb', 0.9),
                                                              ('wc', 0.004)])),
                                        ('fc10', OrderedDict([('epsw', hp_choice('epsw_fc10', [1e-2, 3e-3, 1e-3, 5e-4, 1e-4])),
                                                              ('epsb', hp_choice('epsb_fc10', [2e-2, 6e-3, 2e-3, 5e-4, 2e-4])),
                                                              ('momw', 0.9),
                                                              ('momb', 0.9),
                                                              ('wc', 0.01)])),
                                        ('logprob', OrderedDict([('coeff', 1)]))])



def config_interpretation(layers):
    layers = copy.deepcopy(layers)
    newlayers = OrderedDict([])
    for (l_ind, l) in enumerate(layers):
        print(l_ind, layers[l]['type'])
        if layers[l]['type'] == 'pool' and isinstance(layers[l]['pool'], float):
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
                               ('neuron', 'linear[1, 0.00000001]'),
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
        elif layers[l]['type'] == None:
            if l_ind < len(layers):
                inputs = l['inputs']
                outputs = [_n for _n in layers if l in layers[_n].split(',')]
                for o in outputs:
                    layers[o]['inputs'] = layers[o]['inputs'].replace(l, inputs)
        else:
            newlayers[l] = layers[l]

    return newlayers


def template_func(args):
    return {'layer_def': layer_def_template,
            'learning_params': learning_params_template}
