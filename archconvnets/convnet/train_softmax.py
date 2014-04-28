import numpy.random as nr
import json
import tempfile
import os
import cPickle
from collections import OrderedDict

from .gpumodel import IGPUModel
from .convnet import ConvNet
from .api import odict_to_config


def train_softmax(data_path, train_range, test_range, test_freq, save_freq, epochs, eid,
          initW=0.01, epsw=0.001, epsb=0.001, momw=0.9, momb=0.9,
          db_name='softmax_results', fs_name='softmax_results', gpu_num=None):

    """train (and test) softmax classifier on top of a feature set.  
       
       this makes a specific call to convnet.py so results are stored in a convnet checkpoint db,
       just as with any such convnet training process.   The specific call that is made is to train a single 
       fc layer, followed by a softmax. 
    
       data_path: location of the feature batches (expected to be in the standard archconvnets format, 
           so thus labels are containined as well for each image)
       train_range: valid train range specification, e.g. "0-20"
       test_range: valid test range specificaiton, e.g. "20-40"
       test_freq, save_freq, epochs:  as in convnet.py
       eid: string, experiment_id for storing results
       db_name: db where results are stored 
       fs_name: fs where results are stored
       initW, epsw, epsb, momw, momb: floats.  parameters to the training.
    """

    if isinstance(data_path, str):
        data_path = [data_path]

    X = [cPickle.loads(open(os.path.join(d, 'batches.meta')).read()) for d in data_path]
    nv = sum([x['num_vis'] for x in X])
    nl = len(X[0]['label_names'])
    
    layer_def = OrderedDict([('data', OrderedDict([('type', 'data'), 
                                                   ('dataidx', '0')])), 
                             ('labels', OrderedDict([('type', 'data'),
                                                     ('dataidx', '1')])),
                             ('fc', OrderedDict([('type', 'fcdropo'), 
                                                 ('outputs', '%d' % nl), 
                                                 ('inputs', 'data'),
                                                 ('initw', str(initW)), 
                                                 ('initb', '1'), 
                                                 ('rate', '0.5')])),
                             ('probs', OrderedDict([('type', 'softmax'), 
                                                    ('inputs', 'fc')])),
                             ('logprob', OrderedDict([('type', 'cost.logreg'), 
                                                      ('inputs', 'labels,probs')]))])    
    
    layer_params = OrderedDict([('fc', OrderedDict([('epsw', str(epsw)), 
                                                    ('epsb', str(epsb)), 
                                                    ('momw', str(momw)), 
                                                    ('momb', str(momb)), 
                                                    ('wc', '0.0005')])), 
                                ('logprob', OrderedDict([('coeff', '1')]))])
    
    _, layer_fname = tempfile.mkstemp()
    odict_to_config(layer_def, savepath=layer_fname)
    _, layer_param_fname = tempfile.mkstemp()
    odict_to_config(layer_params, savepath=layer_param_fname)

    
    exp_string = json.dumps(OrderedDict([("experiment_id", eid)]))
    op = ConvNet.get_options_parser()
    oppdict = [('--save-db', '1'),
               ('--save-recent-filters', '0'),
               ('--train-range', train_range),
               ('--test-range', test_range),
               ('--layer-def', layer_fname),
               ('--layer-params', layer_param_fname),               
               ('--conserve-mem', '1'),
               ('--data-provider', 'labeled-data-trans'),
               ('--data-path', "|".join(data_path)),
               ('--test-freq', test_freq),
               ('--saving-freq', save_freq),
               ('--epochs', epochs),
               ('--experiment-data', exp_string),
               ('--img-size', '1'),
               ('--img-channels', '%d' % nv),
               ('--checkpoint-db-name', db_name),
               ("--checkpoint-fs-name", fs_name)]
    if gpu_num is None:
        gpu_num = os.environ.get('BANDIT_GPU', None)
    if gpu_num is not None:
        oppdict.append(('--gpu', gpu_num))

    op, load_dic = IGPUModel.parse_options(op, input_opts=OrderedDict(oppdict), ignore_argv=True)
    nr.seed(0)
    model = ConvNet(op, load_dic)
    try:
        model.start()
    except SystemExit, e:
        if not e.code == 0:
            raise e
