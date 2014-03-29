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


num_filters1 = scope.int(hp_qloguniform('num_filters1',np.log(92), np.log(96), q=16))
filter1_size = scope.int(hp_quniform('filter1_shape', 10, 12, 1))

num_filters2 = scope.int(hp_qloguniform('num_filters2',np.log(250), np.log(256), q=16))
filter2_size = scope.int(hp_quniform('filter2_shape', 4, 6, 1))

num_filters3 = scope.int(hp_qloguniform('num_filters3',np.log(350), np.log(384), q=16))
filter3_size = scope.int(hp_quniform('filter3_shape', 3, 4, 1))

num_filters4 = scope.int(hp_qloguniform('num_filters4',np.log(350), np.log(384), q=16))
filter4_size = scope.int(hp_quniform('filter4_shape', 3, 4, 1))

num_filters5 = scope.int(hp_qloguniform('num_filters5',np.log(350), np.log(384), q=16))
filter5_size = '3'#scope.int(hp_quniform('filter5_shape', 3, 3, 1))

pool1_sizex = '3' #scope.int(hp_quniform('pool1_sizex', 2, 5, 1))
pool1_type = 'max' #hp_choice('pool1_type', ['max', 'avg', hp_uniform('pool_order_1', 1, 4)])

pool2_sizex = '3' #scope.int(hp_quniform('pool2_sizex', 2, 5, 1))
pool2_type = 'max' #hp_choice('pool2_type', ['max', 'avg', hp_uniform('pool_order_2', 1, 4)])

pool3_sizex = '3' #scope.int(hp_quniform('pool3_sizex', 2, 5, 1))
pool3_type = 'max' #hp_choice('pool3_type', ['max', 'avg', hp_uniform('pool_order_3', 1, 4)])

rnorm1_size = '5' #scope.int(hp_quniform('rnorm1_size', 2, 8, 1))
rnorm2_size = '5' #scope.int(hp_quniform('rnorm2_size', 2, 8, 1))
rnorm3_size = '5' #scope.int(hp_quniform('rnorm3_size', 2, 8, 1))


layer_def_template = OrderedDict([('data', OrderedDict([('type', 'data'),
                                   ('dataidx', 0)])),
              ('labels', OrderedDict([('type', 'data'),
                                     ('dataidx', 1)])),
              ('conv1', OrderedDict([('type', 'conv'),
                                    ('inputs', 'data'),
                                    ('channels', 3),
                                    ('filters', num_filters1),
                                    ('padding', 2),
                                    ('stride', 4),
                                    ('filtersize', filter1_size),
                                    ('neuron', 'relu'),
                                    ('initw', hp_choice('initw_conv1', [1e-2])), #('partialsum', 1),
                                    ('sharedbiases', 1)])),
              ('rnorm1', OrderedDict([('type', 'cmrnorm'),
                                                       ('inputs', 'conv1'),
                                                       ('channels', num_filters1),
                                                       ('size', rnorm1_size)])),
              ('pool1', OrderedDict([('type', 'pool'),
                                                        ('pool', pool1_type),
                                                        ('inputs', 'rnorm1'),
                                                        ('start', '0'),
                                                        ('sizex', pool1_sizex),
                                                        ('stride', 2),
                                                        ('outputsx', 0),
                                                        ('channels', num_filters1)])),
              ('conv2', OrderedDict([('type', 'conv'),
                                    ('inputs', 'pool1'), 
                                    ('filters', num_filters2),
                                    ('padding', 2),
                                    ('stride', 1),
                                    ('filtersize', filter2_size),
                                    ('channels', num_filters1),
                                    ('neuron', 'relu'),
                                    ('initw', hp_choice('initw_conv2', [1e-2])), #('partialsum', 1),
                                    ('sharedbiases', 1)])),
              ('rnorm2', OrderedDict([('type', 'cmrnorm'),
                                                       ('inputs', 'conv2'),
                                                       ('channels', num_filters2),
                                                       ('size', rnorm2_size)])),
              ('pool2', OrderedDict([('type', 'pool'),
                                                        ('pool', pool2_type),
                                                        ('inputs', 'rnorm2'),
                                                        ('start', 0),
                                                        ('sizex', pool2_sizex),
                                                        ('stride', 2),
                                                        ('outputsx', 0),
                                                        ('channels', num_filters2)])),
              ('conv3', OrderedDict([('type', 'conv'),
                                             ('inputs', 'pool2'),
                                             ('filters', num_filters3),
                                             ('padding', 2),
                                             ('stride', 1),
                                             ('filtersize', filter3_size),
                                             ('channels', num_filters2),
                                             ('neuron', 'relu'),
                                             ('initw', hp_choice('initw_conv3', [1e-2])), #('partialsum', 1),
                                             ('sharedbiases', 1)])),
             ('conv4', OrderedDict([('type', 'conv'),
                                                 ('inputs', 'conv3'),
                                                 ('filters', num_filters4),
                                                 ('padding', 2),
                                                 ('stride', 1),
                                                 ('filtersize', filter4_size),
                                                 ('channels', num_filters3),
                                                 ('neuron', 'relu'),
                                                 ('initw', hp_choice('initw_conv4',[1e-2])), #('partialsum', 1),
                                                 ('sharedbiases', 1)])),
            ('conv5', OrderedDict([('type', 'conv'),
                                                 ('inputs', 'conv4'),
                                                 ('filters', num_filters5),
                                                 ('padding', 2),
                                                 ('stride', 1),
                                                 ('filtersize', filter5_size),
                                                 ('channels', num_filters4),
                                                 ('neuron', 'relu'),
                                                 ('initw', hp_choice('initw_conv5', [1e-2])), #('partialsum', 1),
                                                 ('sharedbiases', 1)])),
            ('rnorm3', OrderedDict([('type', 'cmrnorm'),
                                                       ('inputs', 'conv5'),
                                                       ('channels', num_filters5),
                                                       ('size', rnorm3_size)])),                      
            ('pool3', OrderedDict([('type', 'pool'),
                                                        ('pool', pool3_type),
                                                        ('inputs', 'rnorm3'),
                                                        ('start', '0'),
                                                        ('sizex', pool3_sizex),
                                                        ('stride', 1),
                                                        ('outputsx', 0),
                                                        ('channels', num_filters5)])),
             ('fc1', OrderedDict([('type', 'fc'),
                                                     ('outputs', scope.int(hp_quniform('num_outputs_fc1', 4024, 4096, 1))),
                                                     ('inputs', 'pool3'),
                                                     ('initw', hp_choice('initw_fc1', [1e-2])),
                                                     ('neuron', 'relu')])),
             ('fc2', OrderedDict([('type', 'fc'),
                                                     ('outputs', scope.int(hp_quniform('num_outputs_fc2', 4024, 4096, 1))),
                                                     ('inputs', 'fc1'),
                                                     ('initw', hp_choice('initw_fc2', [1e-2])),
                                                     ('neuron', 'relu')])),
             ('fc10', OrderedDict([('type', 'fc'),
                                   ('outputs', 999),
                                   ('inputs', 'fc2'),
                                   ('initw', hp_choice('initw_fc10', [1e-2]))])),
             ('probs', OrderedDict([('type', 'softmax'),
                                    ('inputs', 'fc10')])),
             ('logprob', OrderedDict([('type', 'cost.logreg'),
                                      ('inputs', 'labels,probs')]))])


learning_params_template = OrderedDict([('conv1', OrderedDict([('epsw', hp_choice('epsw_conv1', [1e-2])),
                                                               ('epsb', hp_choice('epsb_conv1', [0])),
                                                               ('momw', 0.9),
                                                               ('momb', 0),
                                                               ('wc', hp_choice('wc_conv1', [0.0005]))])),
                                        ('rnorm1', OrderedDict([('scale', hp_choice('rnorm1_scale', [0.00005, 0.0001, 0.0002])),
                                                                ('pow', hp_uniform('rnorm1_pow', 0.6, 0.9))])),
                                        ('conv2', OrderedDict([('epsw', hp_choice('epsw_conv2', [1e-2])),
                                                               ('epsb', hp_choice('epsb_conv2', [0])),
                                                               ('momw', 0.9),
                                                               ('momb', 0),
                                                               ('wc', hp_choice('wc_conv2', [0.0005]))])),
                                        ('rnorm2', OrderedDict([('scale', hp_choice('rnorm2_scale', [0.00005, 0.0001, 0.0002])),
                                                                ('pow', hp_uniform('rnorm2_pow', 0.6, 0.9))])), #('pow', hp_uniform('rnorm2_pow', 0.6, 0.95))])),
                                        ('conv3', OrderedDict([('epsw', hp_choice('epsw_conv3', [1e-2])),
                                                               ('epsb', hp_choice('epsb_conv3', [0])),
                                                               ('momw', 0.9),
                                                               ('momb', 0),
                                                               ('wc', hp_choice('wc_conv3', [0.0005]))])),
                                        ('rnorm3', OrderedDict([('scale', hp_choice('rnorm3_scale', [0.00005, 0.0001, 0.0002])),
                                                                ('pow', hp_uniform('rnorm3_pow', 0.6, 0.9))])), #('pow', hp_uniform('rnorm3_pow', 0.6, 0.95))])),                                
                                        ('conv4', OrderedDict([('epsw', hp_choice('epsw_conv4', [1e-2])),
                                                               ('epsb', hp_choice('epsb_conv4', [0])),
                                                               ('momw', 0.9),
                                                               ('momb', 0),
                                                               ('wc', hp_choice('wc_conv4', [0.0005]))])),
                                        ('conv5', OrderedDict([('epsw', hp_choice('epsw_conv5', [1e-2])),
                                                               ('epsb', hp_choice('epsb_conv5', [0])),
                                                               ('momw', 0.9),
                                                               ('momb', 0),
                                                               ('wc', hp_choice('wc_conv5', [0.0005]))])),
                                        ('fc1', OrderedDict([('epsw', hp_choice('epsw_fc1', [1e-2])),
                                                              ('epsb', hp_choice('epsb_fc1', [0])),
                                                              ('momw', 0.9),
                                                              ('momb', 0.9),
                                                              ('wc', hp_choice('wc_convfc1', [0.0005]))])),
                                        ('fc2', OrderedDict([('epsw', hp_choice('epsw_fc2', [1e-2])),
                                                              ('epsb', hp_choice('epsb_fc2', [0])),
                                                              ('momw', 0.9),
                                                              ('momb', 0.9),
                                                              ('wc', hp_choice('wc_convfc2', [0.0005]))])),
                                        ('fc10', OrderedDict([('epsw', hp_choice('epsw_fc10', [1e-2])),
                                                              ('epsb', hp_choice('epsb_fc10', [1e-2])),
                                                              ('momw', 0.9),
                                                              ('momb', 0.9),
                                                              ('wc', hp_choice('wc_convfc', [0.0005]))])),
                                        ('logprob', OrderedDict([('coeff', 1)]))])



def config_interpretation(layers):
    layers = copy.deepcopy(layers)
    newlayers = OrderedDict([])
    for (l_ind, l) in enumerate(layers):
        #print(l_ind, layers[l]['type'])
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
        elif layers[l]['type'] == 'conv':
            #print 'tttttttttest'
            #print l
            #print layers[l]
            #print layers
            layers[l]['partialsum'] = 9999
            newlayers[l] = layers[l]
        elif layers[l]['type'] == None:
            if l_ind < len(layers):
                inputs = layers[l]['inputs']
                outputs = [_n for _n in layers if l in layers[_n].get('inputs', '').split(',')]
                print('outputs', outputs)
                for o in outputs:
                    layers[o]['inputs'] = layers[o]['inputs'].replace(l, inputs)
        else:
            newlayers[l] = layers[l]
    print('NL', newlayers)
    return newlayers


def template_func(args):
    return {'layer_def': layer_def_template,
            'learning_params': learning_params_template}
