import os
import math
import cPickle
import numpy as np


class Dldata2ConvnetProviderBase(object):
    """Base Object for creating Convnet data provider object from dldata object to

    Params:
        imgs: dldata image array, e.g. result of calling
                    dataset.get_images(preproc)
        metadata: metadata list for single label, e.g. result of
                    dataset.meta[desired key name]
        batch_size: (int) integer batch size
        batch_range: (list of ints, optional) = list of batches to use

    usage:
        >>> imgs = dataset.get_images(preproc=preproc)
        >>> metadata = dataset.meta['category']
        >>> provider = Dldata2ConvnetProviderBase(imgs, metadata, 200)

    """
    def __init__(self, imgs, metadata, batch_size, batch_range=None, init_epoch=1,
                             init_batchnum=None, dp_params=None, test=False):
        if dp_params is None:
            dp_params = {}
        self.dp_params = dp_params
        self.test = test

        self.batch_size = batch_size
        total_batches = int(math.ceil(len(imgs) / float(batch_size)))
        if batch_range is None:
            batch_range = range(1, total_batches + 1)
        assert set(batch_range) <= set(range(1, total_batches + 1))
        self.batch_range = batch_range
        if init_batchnum is None or init_batchnum not in batch_range:
            init_batchnum = batch_range[0]
        self.batch_idx = batch_range.index(init_batchnum)

        self.num_colors = imgs.shape[3] if imgs.ndim == 4 else 1

        self.curr_batchnum = init_batchnum
        self.curr_epoch = init_epoch
        self.data_dic = None

        mshape = imgs.shape[1]
        assert mshape == imgs.shape[2], 'imgs must be square'
        self.imgs = imgs
        self.img_size = mshape

        labels = np.unique(metadata)
        self._num_classes = len(labels)
        self.metadata = np.zeros(len(metadata)).astype(np.single)
        for mind in range(self._num_classes):
            self.metadata[metadata == labels[mind]] = mind

    def get_data_dims(self, idx=0):
        ###what about "if idx == 0 else 1"
        print(idx)
        return (self.imgs.shape[1]**2) * self.num_colors if idx == 0 else 1

    def get_next_batch(self):
        if self.data_dic is None or len(self.batch_range) > 1:
            self.data_dic = self.get_batch(self.curr_batchnum)
        epoch, batchnum = self.curr_epoch, self.curr_batchnum
        self.advance_batch()
        return epoch, batchnum, [self.data_dic['data'], self.data_dic['labels']]

    def get_batch(self, batch_num):
        bn = batch_num - 1
        data = self.imgs[bn * self.batch_size: (bn+1) * self.batch_size]
        data = np.asarray(data, dtype=np.single)

        mshape = data.shape[1]
        new_s = (data.shape[0], mshape **2)
        if data.ndim == 4:
            nc = self.num_colors
            data = np.column_stack([data[:, :, :, i].reshape(new_s) for i in range(nc)]).T
        else:
            data = data.reshape(new_s).T

        metadata = self.metadata
        labels = metadata[bn * self.batch_size: (bn+1) * self.batch_size]
        labels = labels.reshape((1, len(labels)))

        return {'data': data, 'labels': labels}

    def advance_batch(self):
        self.batch_idx = self.get_next_batch_idx()
        self.curr_batchnum = self.batch_range[self.batch_idx]
        if self.batch_idx == 0: # we wrapped
            self.curr_epoch += 1

    def get_next_batch_idx(self):
        return (self.batch_idx + 1) % len(self.batch_range)

    def get_next_batch_num(self):
        return self.batch_range[self.get_next_batch_idx()]

    def get_num_classes(self):
        return self._num_classes
    
    def get_out_img_size(self):
        return self.img_size

    def get_out_img_depth(self):
        return self.num_colors

import skdata.cifar10 as cf10

class CIFAR10TestProvider(Dldata2ConvnetProviderBase):
    """for test purposes ONLY
    """
    def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params=None, test=False):
        dataset = cf10.dataset.CIFAR10()
        meta = dataset.meta
        meta = np.array([x['label'] for x in meta])
        imgs = dataset._pixels
        batch_size = 10000
        Dldata2ConvnetProviderBase.__init__(self, imgs, meta, batch_size, 
                                            batch_range=batch_range, 
                                            init_epoch=init_epoch, 
                                            init_batchnum=init_batchnum, 
                                            dp_params=dp_params,
                                            test=test)

        bmfile = os.path.join(os.path.split(__file__)[0], 
               'data', 'cifar-10-py-colmajor', 'batches.meta')
        self.batches_meta = cPickle.load(open(bmfile))

    def get_next_batch(self):
        bn = self.curr_batchnum
        a, b, c = Dldata2ConvnetProviderBase.get_next_batch(self)
        c[0] = c[0] - self.batches_meta['data_mean']
        return a, b, c
