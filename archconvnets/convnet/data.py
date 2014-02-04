# Copyright (c) 2011, Alex Krizhevsky (akrizhevsky@gmail.com)
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# - Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# 
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import math
import importlib
import numpy as n
from numpy.random import randn, rand, random_integers
import os
from util import *

BATCH_META_FILE = "batches.meta"

class DataProvider:
    BATCH_REGEX = re.compile('^data_batch_(\d+)(\.\d+)?$')
    def __init__(self, data_dir, batch_range=None, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        if batch_range == None:
            batch_range = DataProvider.get_batch_nums(data_dir)
        if init_batchnum is None or init_batchnum not in batch_range:
            init_batchnum = batch_range[0]

        self.data_dir = data_dir
        self.batch_range = batch_range
        self.curr_epoch = init_epoch
        self.curr_batchnum = init_batchnum
        self.dp_params = dp_params
        self.batch_meta = self.get_batch_meta(data_dir)
        self.data_dic = None
        self.test = test
        self.batch_idx = batch_range.index(init_batchnum)

    def get_next_batch(self):
        if self.data_dic is None or len(self.batch_range) > 1:
            self.data_dic = self.get_batch(self.curr_batchnum)
        epoch, batchnum = self.curr_epoch, self.curr_batchnum
        self.advance_batch()

        return epoch, batchnum, self.data_dic
    
    def __add_subbatch(self, batch_num, sub_batchnum, batch_dic):
        subbatch_path = "%s.%d" % (os.path.join(self.data_dir, self.get_data_file_name(batch_num)), sub_batchnum)
        if os.path.exists(subbatch_path):
            sub_dic = unpickle(subbatch_path)
            self._join_batches(batch_dic, sub_dic)
        else:
            raise IndexError("Sub-batch %d.%d does not exist in %s" % (batch_num,sub_batchnum, self.data_dir))
        
    def _join_batches(self, main_batch, sub_batch):
        main_batch['data'] = n.r_[main_batch['data'], sub_batch['data']]
        
    def get_batch(self, batch_num):
        if os.path.exists(self.get_data_file_name(batch_num) + '.1'): # batch in sub-batches
            dic = unpickle(self.get_data_file_name(batch_num) + '.1')
            sb_idx = 2
            while True:
                try:
                    self.__add_subbatch(batch_num, sb_idx, dic)
                    sb_idx += 1
                except IndexError:
                    break
        else:
            dic = unpickle(self.get_data_file_name(batch_num))
        return dic
    
    def get_data_dims(self):
        return self.batch_meta['num_vis']
    
    def advance_batch(self):
        self.batch_idx = self.get_next_batch_idx()
        self.curr_batchnum = self.batch_range[self.batch_idx]
        if self.batch_idx == 0: # we wrapped
            self.curr_epoch += 1
            
    def get_next_batch_idx(self):
        return (self.batch_idx + 1) % len(self.batch_range)
    
    def get_next_batch_num(self):
        return self.batch_range[self.get_next_batch_idx()]
    
    # get filename of current batch
    def get_data_file_name(self, batchnum=None):
        if batchnum is None:
            batchnum = self.curr_batchnum
        return os.path.join(self.data_dir, 'data_batch_%d' % batchnum)
    
    @classmethod
    def get_instance(cls, data_dir, 
            img_size, num_colors,  # options i've add to cifar data provider
            batch_range=None, init_epoch=1, init_batchnum=None, type="default", dp_params={}, test=False):
        # why the fuck can't i reference DataProvider in the original definition?
        #cls.dp_classes['default'] = DataProvider
        type = type or DataProvider.get_batch_meta(data_dir)['dp_type'] # allow data to decide data provider
        if type.startswith("dummy-"):
            name = "-".join(type.split('-')[:-1]) + "-n"
            if name not in dp_types:
                raise DataProviderException("No such data provider: %s" % type)
            _class = dp_classes[name]
            dims = int(type.split('-')[-1])
            return _class(dims)
        elif type in dp_types:
            if img_size == 0:
                _class = dp_classes[type]
                return _class(data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
            else :
                _class = dp_classes[type]
                return _class(data_dir, img_size, num_colors,
                        batch_range, init_epoch, init_batchnum, dp_params, test)
        
        raise DataProviderException("No such data provider: %s" % type)
    
    @classmethod
    def register_data_provider(cls, name, desc, _class):
        if name in dp_types:
            raise DataProviderException("Data provider %s already registered" % name)
        dp_types[name] = desc
        dp_classes[name] = _class
        
    @staticmethod
    def get_batch_meta(data_dir):
        return unpickle(os.path.join(data_dir, BATCH_META_FILE))
    
    @staticmethod
    def get_batch_filenames(srcdir):
        return sorted([f for f in os.listdir(srcdir) if DataProvider.BATCH_REGEX.match(f)], key=alphanum_key)
    
    @staticmethod
    def get_batch_nums(srcdir):
        names = DataProvider.get_batch_filenames(srcdir)
        return sorted(list(set(int(DataProvider.BATCH_REGEX.match(n).group(1)) for n in names)))
        
    @staticmethod
    def get_num_batches(srcdir):
        return len(DataProvider.get_batch_nums(srcdir))
    
class DummyDataProvider(DataProvider):
    def __init__(self, data_dim):
        #self.data_dim = data_dim
        self.batch_range = [1]
        self.batch_meta = {'num_vis': data_dim, 'data_in_rows':True}
        self.curr_epoch = 1
        self.curr_batchnum = 1
        self.batch_idx = 0
        
    def get_next_batch(self):
        epoch,  batchnum = self.curr_epoch, self.curr_batchnum
        self.advance_batch()
        data = rand(512, self.get_data_dims()).astype(n.single)
        return self.curr_epoch, self.curr_batchnum, {'data':data}

    
class LabeledDummyDataProvider(DummyDataProvider):
    def __init__(self, data_dim, num_classes=10, num_cases=512):
        #self.data_dim = data_dim
        self.batch_range = [1]
        self.batch_meta = {'num_vis': data_dim,
                           'label_names': [str(x) for x in range(num_classes)],
                           'data_in_rows':True}
        self.num_cases = num_cases
        self.num_classes = num_classes
        self.curr_epoch = 1
        self.curr_batchnum = 1
        self.batch_idx=0
        
    def get_num_classes(self):
        return self.num_classes
    
    def get_next_batch(self):
        epoch,  batchnum = self.curr_epoch, self.curr_batchnum
        self.advance_batch()
        data = rand(self.num_cases, self.get_data_dims()).astype(n.single) # <--changed to rand
        labels = n.require(n.c_[random_integers(0,self.num_classes-1,self.num_cases)], requirements='C', dtype=n.single)

        return self.curr_epoch, self.curr_batchnum, {'data':data, 'labels':labels}

class MemoryDataProvider(DataProvider):
    def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params=None, test=False):
        DataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        self.data_dic = []
        for i in self.batch_range:
            self.data_dic += [self.get_batch(i)]
    
    def get_next_batch(self):
        epoch, batchnum = self.curr_epoch, self.curr_batchnum
        self.advance_batch()
        return epoch, batchnum, self.data_dic[batchnum - self.batch_range[0]]


class LabeledDataProvider(DataProvider):   
    def __init__(self, data_dir, batch_range=None, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        DataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        
    def get_num_classes(self):
        return len(self.batch_meta['label_names'])
    

class LabeledMemoryDataProvider(LabeledDataProvider):
    def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        LabeledDataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        self.data_dic = []
        for i in batch_range:
            self.data_dic += [unpickle(self.get_data_file_name(i))]
            self.data_dic[-1]["labels"] = n.c_[n.require(self.data_dic[-1]['labels'], dtype=n.single)]
            
    def get_next_batch(self):
        epoch, batchnum = self.curr_epoch, self.curr_batchnum
        self.advance_batch()
        bidx = batchnum - self.batch_range[0]
        return epoch, batchnum, self.data_dic[bidx]

        
def dldata_to_convnet_reformatting(stims, lbls):
    img_sz = stims.shape[1]
    batch_size = stims.shape[0]
    if stims.ndim == 3:
        new_s = (batch_size, img_sz**2)
        stims = stims.reshape(new_s).T
    else:
        assert stims.ndim == 4
        nc = stims.shape[3]
        new_s = (nc * (img_sz**2), batch_size)
        stims = stims.transpose([3, 1, 2, 0]).reshape(new_s)    
        
    assert lbls.ndim == 1
    labels = lbls.reshape((1, lbls.shape[0]))
    return {'data': stims, 'labels': labels}


class DLDataProvider(LabeledDataProvider):

    def __init__(self, data_dir, batch_range, init_epoch=1,
                        init_batchnum=None, dp_params=None, test=False):
                        
        #load dataset and meta
        self.replace_label = dp_params.get('replace_label', False)
        modulename, attrname = dp_params['dataset_name']
        module = importlib.import_module(modulename)
        dataset_obj = getattr(module, attrname)
        dataset_data = dp_params.get('dataset_data', None)
        if dataset_data is not None:
            dset = dataset_obj(data=dataset_data)
        else:
            dset = dataset_obj()
        meta = self.meta = dset.meta   
        mlen = len(meta)
        self.dp_params = dp_params
        
        #compute number of batches
        mlen = len(meta)   
        batch_size = dp_params['batch_size'] 
        num_batches = self.num_batches = int(math.ceil(mlen / float(batch_size)))
 
        batch_regex = re.compile('data_batch_([\d]+)')
        imgs_mean = None
        existing_batches = []
        isf = 0
        if os.path.exists(data_dir):
            _L = os.listdir(data_dir)
            existing_batches = [int(batch_regex.match(_l).groups()[0]) for _l in _L if batch_regex.match(_l)]
            existing_batches.sort()
            metafile = os.path.join(data_dir, 'batches.meta')
            if existing_batches:
                assert os.path.exists(metafile), 'Batches found but no metafile %s' % metafile
            if os.path.exists(metafile):
                bmeta = cPickle.load(open(metafile))
                ebatches = bmeta['existing_batches']
                imgs_mean = bmeta['data_mean']
                isf = bmeta['images_so_far']
            else:
                ebatches = []
            assert existing_batches == ebatches, ('Expected batches', ebatches, 'found batches', existing_batches)
            needed_batches = [_b for _b in batch_range if _b not in existing_batches]
            if existing_batches:
                print('Found batches: ', existing_batches)
                print('Batches needed: ', needed_batches)
        else:
            print('data_dir %s does not exist, creating' % data_dir)
            needed_batches = batch_range[:]
            os.makedirs(data_dir)
        
        if needed_batches or self.replace_label:
            indset = self.indset = self.get_indset()
            metacol = self.metacol = self.get_metacol()
        
        if needed_batches:
            #get stimarray (may be lazyarray)
            #something about appearing to require uint8??
            #dp_params['preproc']['dtype'] = 'uint8'    #or assertion?
            stimarray = dset.get_images(preproc=dp_params['preproc'])
 
            #actually write out batches, while tallying img mean
            for bnum, inds in enumerate(indset):
                if bnum not in needed_batches:
                    continue
                print('Creating batch %d' % bnum)
                #get stimuli and put in the required format
                stims = n.asarray(stimarray[inds])
                if 'float' in repr(stims.dtype):
                    stims = n.uint8(n.round(255 * stims))
                lbls = metacol[inds]
                d = dldata_to_convnet_reformatting(stims, lbls)
                
                #add to the mean
                if imgs_mean is None:
                    imgs_mean = n.zeros((d['data'].shape[0],))
                dlen = d['data'].shape[0]
                fr = isf / (isf + float(dlen))
                imgs_mean *= fr
                imgs_mean += (1 - fr) * d['data'].mean(axis=1)
                isf += dlen
                
                #write out batch
                outdict = {'batch_label': 'batch_%d' % bnum,
                           'labels': d['labels'], 
                           'data': d['data'], 
                      #'filenames': n.asarray(dataset.meta[img_inds]['filename']).tolist()    ####need this????
                           }
                outpath = os.path.join(data_dir, 'data_batch_%d' % bnum)
                with open(outpath, 'w') as _f:
                    cPickle.dump(outdict, _f)
                    
            #write out batches.meta
            existing_batches += needed_batches
            existing_batches.sort()
            outdict = {'num_cases_per_batch': batch_size, 
                       'label_names': self.labels_unique, 
                       'num_vis': d['data'].shape[0], 
                       'data_mean': imgs_mean,
                       'existing_batches': existing_batches,
                       'images_so_far': isf}   
            with open(os.path.join(data_dir, 'batches.meta'), 'w') as _f:
                cPickle.dump(outdict, _f)
       
        LabeledDataProvider.__init__(self, data_dir, batch_range, 
                                 init_epoch, init_batchnum, dp_params, test)

        if self.replace_label:
            self.batch_meta['label_names'] = self.labels_unique

    def get_next_batch(self):
        epoch, batchnum, d = LabeledDataProvider.get_next_batch(self)            
        d['data'] = n.require(d['data'], requirements='C')  
        d['labels'] = n.c_[n.require(d['labels'], dtype=n.single)]
        return epoch, batchnum, d
        
    def get_batch(self, batch_num):
        dic = LabeledDataProvider.get_batch(self, batch_num)
        if self.replace_label:
            metacol = self.metacol
            indset = self.indset            
            lbls = metacol[indset[batch_num]]
            assert lbls.ndim == 1
            labels = lbls.reshape((1, lbls.shape[0]))
            dic['labels'] = labels
        return  dic

    def get_metacol(self):
        meta = self.meta
        mlen = len(meta)
        #format relevant metadata column into integer list if needed
        metacol = meta[self.dp_params['meta_attribute']][:]
        try:
            metacol + 1
            labels_unique = None
        except TypeError:
            labels_unique = self.labels_unique = n.unique(metacol)
            labels = n.zeros((mlen, ), dtype='int')
            for label in range(len(labels_unique)):
                labels[metacol == labels_unique[label]] = label
            metacol = labels
        return metacol

    def get_indset(self):
        dp_params = self.dp_params
        perm_type = dp_params.get('perm_type')
        perm_type = dp_params.get('perm_type')
        num_batches = self.num_batches
        batch_size = dp_params['batch_size']
        meta = self.meta
        
        if perm_type is not None:
            mlen = len(self.meta)
            if perm_type == 'random':
                perm_seed = dp_params.get('perm_seed', 0)
                rng = n.random.RandomState(seed=perm_seed)
                perm = rng.permutation(mlen)
                indset = [perm[batch_size * bidx: batch_size * (bidx + 1)] for bidx in range(num_batches)]
            elif perm_type == 'ordered_random':
                perm_seed = dp_params.get('perm_seed', 0)
                rng = n.random.RandomState(seed=perm_seed)
                perm = rng.permutation(mlen)
                submeta = meta[dp_params['perm_order']].copy()
                submeta = submeta[perm]
                s = submeta.argsort(order=dp_params['perm_order'])
                new_perm = perm[s]
                indset = [new_perm[batch_size * bidx: batch_size * (bidx + 1)] for bidx in range(num_batches)]
            else:
                raise ValueError, 'Unknown permutation type.'
        else:
            indset = [slice(batch_size * bidx, batch_size * (bidx + 1)) 
                       for bidx in range(num_batches)]
        return indset

    
dp_types = {"default": "The default data provider; loads one batch into memory at a time",
            "memory": "Loads the entire dataset into memory",
            "labeled": "Returns data and labels (used by classifiers)",
            "labeled-memory": "Combination labeled + memory",
            "dummy-n": "Dummy data provider for n-dimensional data",
            "dummy-labeled-n": "Labeled dummy data provider for n-dimensional data"}
dp_classes = {"default": DataProvider,
              "memory": MemoryDataProvider,
              "labeled": LabeledDataProvider,
              "labeled-memory": LabeledMemoryDataProvider,
              "dummy-n": DummyDataProvider,
              "dummy-labeled-n": LabeledDummyDataProvider}
    
class DataProviderException(Exception):
    pass

