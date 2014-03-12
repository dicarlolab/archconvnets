__author__ = 'headradio'


import dldata.stimulus_sets.hvm as hvm
import re
import os
import cPickle
import numpy as np

dataset = hvm.HvMWithDiscfade()
neural_data = dataset.neuronal_features[:, dataset.IT_NEURONS]
meta = dataset.meta
batch_regex = re.compile('data_batch_([\d]+)')
imgs_mean = None
existing_batches = []
batch_path = '/export/imgnet_storage_full/ardila/hvm_batches2'
_L = os.listdir(batch_path)
existing_batches = [int(batch_regex.match(_l).groups()[0]) for _l in _L if batch_regex.match(_l)]
for batch in existing_batches:
    print batch
    path = os.path.join(batch_path, 'data_batch_'+str(batch))
    vectors = []
    print path
    data_dic = cPickle.load(open(path, 'rb'))
    ids = data_dic['ids']
    for image_id in ids:
        vectors.append(list(np.ravel(neural_data[meta['id'] == image_id])))
    vectors = np.column_stack(vectors)
    data_dic['vectors'] = [vectors]
    cPickle.dump(data_dic, open(path, 'wb'))
batch_meta = cPickle.load(open(os.path.join(batch_path, 'batches.meta'), 'rb'))
batch_meta['vector_dims'] = [v.shape[0] for v in data_dic['vectors']]
batch_meta = cPickle.dump(batch_meta, open(os.path.join(batch_path, 'batches.meta'), 'wb'))
