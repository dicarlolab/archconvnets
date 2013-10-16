import os
import math
import cPickle
import numpy as np
import hashlib
import copy
from dldata.stimulus_sets.dataset_templates import ImageLoaderPreprocesser


def get_id(l):
    return hashlib.sha1(repr(l)).hexdigest()


class Dldata2ConvnetProviderBase(object):
    """Base Object for creating Convnet data provider object from dldata object. Stores one batch in memory at a time.

    Params:
        dataset: dldata object
        preproc: preprocessing dict used to specify how the images are preprocessed before caching.
        metacol: used to generate metadata list for single label, e.g.
                    dataset.meta[metacol]
        batch_size: (int) integer batch size

        you can also circumvent this and pass images and metadata lists directly using the name arguments
        metadata and imgs, but this is discouraged.


        batch_range: (list of ints, optional) = list of batches to use

    usage:
        >>> dataset =
        >>> metacol = 'category'
        >>> provider = Dldata2ConvnetProviderBase(dataset, preproc, metacol, 200)

    """

    def __init__(self, batch_size=None, imgs=None, metadata=None, dataset=None, preproc=None,
                 metacol=None, batch_range=None, init_epoch=1, init_batchnum=None,
                 dp_params=None, test=False):

        if dp_params is None:
            dp_params = {}
        self.dp_params = dp_params
        self.test = test

        self.dataset = dataset
        self.preproc = preproc
        self.metacol = metacol
        self.postproc = copy.deepcopy(preproc)
        self.eig_mod = dp_params['eig_mod']
        self.border_size = dp_params['crop_border']
        if self.eig_mod is not None:
            if self.dataset is None:
                raise 'PCA intensity modulation is only implemented for dldatasets. Convert your images and meta to a' \
                      'dldata object (see wiki or documentation in dataset_templates)'
            eigval, eigvec = self.dataset.get_pixel_eigenvalues_and_eigenvectors()
            sd = self.eig_mod
            self.postproc['eig_mod'] = (eigval, eigvec, sd)

        if self.eig_mod is not None:
            self.post_proc['crop_rand'] = self.border_size  # need to figure out what this does when the option is not
            #passed

        if imgs is None:
            imgs = dataset.get_images(preproc)

        if metadata is None:
            metadata = dataset.meta[metacol]

        self.batch_size = batch_size
        total_batches = int(math.ceil(len(imgs) / float(batch_size)))
        if batch_range is None:
            batch_range = range(1, total_batches + 1)
        assert set(batch_range) <= set(range(1, total_batches + 1)), (batch_range, total_batches)
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
        return (self.imgs.shape[1] ** 2) * self.num_colors if idx == 0 else 1

    def get_next_batch(self):
        if self.data_dic is None or len(self.batch_range) > 1:
            self.data_dic = self.get_batch(self.curr_batchnum)
        self.data_dic['data'] = ImageLoaderPreprocesser(self.postproc)(self.data_dic['data'])
        epoch, batchnum = self.curr_epoch, self.curr_batchnum
        self.advance_batch()
        return epoch, batchnum, [self.data_dic['data'], self.data_dic['labels']]

    def get_batch(self, batch_num):
        bn = batch_num - 1

        preproc = self.preproc
        batch_size = self.batch_size
        metacol = self.metacol

        if self.dataset:
            cache = True
            batchdir = self.dataset.home('batch_caches', get_id((preproc, batch_size, metacol)))
            if not os.path.isdir(batchdir):
                os.makedirs(batchdir)
            batchfile = os.path.join(batchdir, 'batch_%d' % batch_num)
        else:
            cache = False

        if not cache or not os.path.exists(batchfile):
            data = self.imgs[bn * self.batch_size: (bn + 1) * self.batch_size]
            data = np.asarray(data, dtype=np.single)

            mshape = data.shape[1]
            new_s = (data.shape[0], mshape ** 2)
            if data.ndim == 4:
                nc = self.num_colors
                data = np.column_stack([data[:, :, :, i].reshape(new_s) for i in range(nc)]).T
            else:
                data = data.reshape(new_s).T

            metadata = self.metadata
            labels = metadata[bn * self.batch_size: (bn + 1) * self.batch_size]
            labels = labels.reshape((1, len(labels)))
            batchval = {'data': data, 'labels': labels}
            if cache:
                with open(batchfile, 'wb') as _f:
                    cPickle.dump(batchval, _f)
            return batchval
        else:
            print('loading from cache %s' % batchfile)
            return cPickle.loads(open(batchfile).read())

    def advance_batch(self):
        self.batch_idx = self.get_next_batch_idx()
        self.curr_batchnum = self.batch_range[self.batch_idx]
        if self.batch_idx == 0:  # we wrapped
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
        Dldata2ConvnetProviderBase.__init__(self, imgs=imgs, metadata=meta, batch_size=batch_size,
                                            batch_range=batch_range,
                                            init_epoch=init_epoch,
                                            init_batchnum=init_batchnum,
                                            dp_params=dp_params,
                                            test=test)

        bmfile = os.path.join(os.path.split(__file__)[0],
                              'data', 'cifar-10-py-colmajor', 'batches.meta')
        self.batches_meta = cPickle.load(open(bmfile))

    def get_next_batch(self):
        a, b, c = Dldata2ConvnetProviderBase.get_next_batch(self)
        c[0] = c[0] - self.batches_meta['data_mean']
        return a, b, c


class HVMCategoryProvider32x32(Dldata2ConvnetProviderBase):
    """hvm provider
    """

    def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params=None, test=False):
        import dldata.stimulus_sets.hvm as hvm

        dataset = hvm.HvMWithDiscfade()
        metacol = 'category'
        preproc = {'size': (32, 32, 3), 'dtype': 'float32', 'global_normalize': False}
        batch_size = 10000
        Dldata2ConvnetProviderBase.__init__(self, dataset=dataset, preproc=preproc,
                                            metacol=metacol, batch_size=batch_size,
                                            batch_range=batch_range,
                                            init_epoch=init_epoch,
                                            init_batchnum=init_batchnum,
                                            dp_params=dp_params,
                                            test=test)


class CIFAR10TestGrayscaleProvider(Dldata2ConvnetProviderBase):
    """for test purposes ONLY
    """

    def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params=None, test=False):
        dataset = cf10.dataset.CIFAR10()
        meta = dataset.meta
        meta = np.array([x['label'] for x in meta])
        imgs = dataset._pixels[:, :, :, 0]
        batch_size = 10000
        Dldata2ConvnetProviderBase.__init__(self, imgs=imgs, metadata=meta, batch_size=batch_size,
                                            batch_range=batch_range,
                                            init_epoch=init_epoch,
                                            init_batchnum=init_batchnum,
                                            dp_params=dp_params,
                                            test=test)


class CIFARHVMTEST(Dldata2ConvnetProviderBase):
    """JUST FOR TEST PURPOSES
    """

    def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params=None, test=False):
        import dldata.stimulus_sets.hvm as hvm

        dataset = hvm.HvMWithDiscfade()
        meta = dataset.meta['category']
        imgs = dataset.get_images({'dtype': 'float32', 'size': (32, 32, 3), 'normalize': False})

        dataset = cf10.dataset.CIFAR10()
        meta1 = dataset.meta
        meta1 = np.array([x['label'] for x in meta1])
        imgs1 = dataset._pixels

        meta = np.concatenate([meta1, meta1[:len(meta)]])
        imgs = np.concatenate([imgs1, imgs[:]])
        batch_size = 10000
        Dldata2ConvnetProviderBase.__init__(self, imgs=imgs, metadata=meta, batch_size=batch_size,
                                            batch_range=batch_range,
                                            init_epoch=init_epoch,
                                            init_batchnum=init_batchnum,
                                            dp_params=dp_params,
                                            test=test)


class ImagenetPixelHardSynsets2013ChallengeTop40Provider(Dldata2ConvnetProviderBase):
    """hvm provider
    """

    def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params=None, test=False):
        import imagenet.dldatasets

        dataset = imagenet.dldatasets.PixelHardSynsets2013ChallengeTop40Screenset()
        metacol = 'synset'
        preproc = {'resize_to': (128, 128), 'dtype': 'float32', 'mode': 'RGB',
                   'normalize': False, 'mask': None, 'crop': None}
        batch_size = 1000
        Dldata2ConvnetProviderBase.__init__(self, dataset=dataset, preproc=preproc,
                                            metacol=metacol, batch_size=batch_size,
                                            batch_range=batch_range,
                                            init_epoch=init_epoch,
                                            init_batchnum=init_batchnum,
                                            dp_params=dp_params,
                                            test=test)
