import math
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
                                               init_batchnum=None):

        self.batch_size = batch_size
        total_batches = int(math.ceil(len(imgs) / float(batch_size)))
        if batch_range is None:
            batch_range = range(total_batches)
        assert set(batch_range) <= set(range(total_batches))
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

        labels = np.unique(metadata)
        self.num_labels = len(labels)
        self.metadata = np.zeros(len(metadata)).astype(np.single)
        for mind in range(self.num_labels):
            self.metadata[metadata == labels[mind]] = mind

    def get_data_dims(self, idx=0):
        ###what about "if idx == 0 else 1"
        return (self.imgs.shape[1]**2) * self.num_colors

    def get_next_batch(self):
        if self.data_dic is None or len(self.batch_range) > 1:
            self.data_dic = self.get_batch(self.curr_batchnum)
        epoch, batchnum = self.curr_epoch, self.curr_batchnum
        self.advance_batch()
        return epoch, batchnum, [self.data_dic['data'], self.data_dic['labels']]

    def get_batch(self, batch_num):
        data = self.imgs[batch_num * self.batch_size: (batch_num+1) * self.batch_size]
        data = np.asarray(data, dtype=np.single)

        mshape = data.shape[1]
        new_s = (data.shape[0], mshape **2)
        if data.ndim == 4:
            nc = self.num_colors
            data = np.column_stack([data[:, :, :, i].reshape(new_s) for i in range(nc)]).T
        else:
            data = data.reshape(new_s).T

        metadata = self.metadata
        labels = metadata[batch_num * self.batch_size: (batch_num+1) * self.batch_size]
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

