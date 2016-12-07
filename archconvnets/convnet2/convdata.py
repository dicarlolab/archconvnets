# Copyright 2014 Google Inc. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#    http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from python_util.data import *
import numpy.random as nr
import numpy as n
import random as r
from time import time
from threading import Thread
from math import sqrt
import sys
#from matplotlib import pylab as pl
from PIL import Image
from StringIO import StringIO
from time import time
import itertools as it
    
class JPEGBatchLoaderThread(Thread):
    def __init__(self, dp, batch_num, label_offset, list_out):
        Thread.__init__(self)
        self.list_out = list_out
        self.label_offset = label_offset
        self.dp = dp
        self.batch_num = batch_num
        
    @staticmethod
    def load_jpeg_batch(rawdics, dp, label_offset):
        if type(rawdics) != list:
            rawdics = [rawdics]
        nc_total = sum(len(r['data']) for r in rawdics)

        jpeg_strs = list(it.chain.from_iterable(rd['data'] for rd in rawdics))
        labels = list(it.chain.from_iterable(rd['labels'] for rd in rawdics))
        
        img_mat = n.empty((nc_total * dp.data_mult, dp.inner_pixels * dp.num_colors), dtype=n.float32)
        lab_mat = n.zeros((nc_total, dp.get_num_classes()), dtype=n.float32)
        dp.convnet.libmodel.decodeJpeg(jpeg_strs, img_mat, dp.img_size, dp.inner_size, dp.test, dp.multiview)
        lab_vec = n.tile(n.asarray([(l[nr.randint(len(l))] if len(l) > 0 else -1) + label_offset for l in labels], dtype=n.single).reshape((nc_total, 1)), (dp.data_mult,1))
        for c in xrange(nc_total):
            lab_mat[c, [z + label_offset for z in labels[c]]] = 1
        lab_mat = n.tile(lab_mat, (dp.data_mult, 1))
        

        return {'data': img_mat[:nc_total * dp.data_mult,:],
                'labvec': lab_vec[:nc_total * dp.data_mult,:],
                'labmat': lab_mat[:nc_total * dp.data_mult,:]}
    
    def run(self):
        rawdics = self.dp.get_batch(self.batch_num)
        p = JPEGBatchLoaderThread.load_jpeg_batch(rawdics,
                                                  self.dp,
                                                  self.label_offset)
        self.list_out.append(p)
        
class ColorNoiseMakerThread(Thread):
    def __init__(self, pca_stdevs, pca_vecs, num_noise, list_out):
        Thread.__init__(self)
        self.pca_stdevs, self.pca_vecs = pca_stdevs, pca_vecs
        self.num_noise = num_noise
        self.list_out = list_out
        
    def run(self):
        noise = n.dot(nr.randn(self.num_noise, 3).astype(n.single) * self.pca_stdevs.T, self.pca_vecs.T)
        self.list_out.append(noise)

class ImageDataProvider(LabeledDataProvider):
    def __init__(self, data_dir, batch_range=None, init_epoch=1, init_batchnum=None, dp_params=None, test=False):
        LabeledDataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        self.data_mean = self.batch_meta['data_mean'].astype(n.single)
        self.color_eig = self.batch_meta['color_pca'][1].astype(n.single)
        self.color_stdevs = n.c_[self.batch_meta['color_pca'][0].astype(n.single)]
        self.color_noise_coeff = dp_params['color_noise']
        self.num_colors = 3
        self.img_size = int(sqrt(self.batch_meta['num_vis'] / self.num_colors))
        self.mini = dp_params['minibatch_size']
        self.inner_size = dp_params['inner_size'] if dp_params['inner_size'] > 0 else self.img_size
        self.inner_pixels = self.inner_size **2
        self.border_size = (self.img_size - self.inner_size) / 2
        self.multiview = dp_params['multiview_test'] and test
        self.num_views = 5*2
        self.data_mult = self.num_views if self.multiview else 1
        self.batch_size = self.batch_meta['batch_size']
        self.label_offset = 0 if 'label_offset' not in self.batch_meta else self.batch_meta['label_offset']
        self.scalar_mean = dp_params['scalar_mean'] 
        # Maintain pointers to previously-returned data matrices so they don't get garbage collected.
        self.data = [None, None] # These are pointers to previously-returned data matrices

        self.loader_thread, self.color_noise_thread = None, None
        self.convnet = dp_params['convnet']
            
        self.num_noise = self.batch_size
        self.batches_generated, self.loaders_started = 0, 0
        self.data_mean_crop = self.data_mean.reshape((self.num_colors,self.img_size,self.img_size))[:,self.border_size:self.border_size+self.inner_size,self.border_size:self.border_size+self.inner_size].reshape((1,3*self.inner_size**2))

        if self.scalar_mean >= 0:
            self.data_mean_crop = self.scalar_mean
            
    def showimg(self, img):
        from matplotlib import pylab as pl
        pixels = img.shape[0] / 3
        size = int(sqrt(pixels))
        img = img.reshape((3,size,size)).swapaxes(0,2).swapaxes(0,1)
        pl.imshow(img, interpolation='nearest')
        pl.show()
            
    def get_data_dims(self, idx=0):
        if idx == 0:
            return self.inner_size**2 * 3
        if idx == 2:
            return self.get_num_classes()
        return 1

    def start_loader(self, batch_idx):
        self.load_data = []
        self.loader_thread = JPEGBatchLoaderThread(self,
                                                   self.batch_range[batch_idx],
                                                   self.label_offset,
                                                   self.load_data)
        self.loader_thread.start()
        
    def start_color_noise_maker(self):
        color_noise_list = []
        self.color_noise_thread = ColorNoiseMakerThread(self.color_stdevs, self.color_eig, self.num_noise, color_noise_list)
        self.color_noise_thread.start()
        return color_noise_list

    def set_labels(self, datadic):
        pass
    
    def get_data_from_loader(self):
        if self.loader_thread is None:
            self.start_loader(self.batch_idx)
            self.loader_thread.join()
            self.data[self.d_idx] = self.load_data[0]

            self.start_loader(self.get_next_batch_idx())
        else:
            # Set the argument to join to 0 to re-enable batch reuse
            self.loader_thread.join()
            if not self.loader_thread.is_alive():
                self.data[self.d_idx] = self.load_data[0]
                self.start_loader(self.get_next_batch_idx())
            #else:
            #    print "Re-using batch"
        self.advance_batch()
    
    def add_color_noise(self):
        # At this point the data already has 0 mean.
        # So I'm going to add noise to it, but I'm also going to scale down
        # the original data. This is so that the overall scale of the training
        # data doesn't become too different from the test data.

        s = self.data[self.d_idx]['data'].shape
        cropped_size = self.get_data_dims(0) / 3
        ncases = s[0]

        if self.color_noise_thread is None:
            self.color_noise_list = self.start_color_noise_maker()
            self.color_noise_thread.join()
            self.color_noise = self.color_noise_list[0]
            self.color_noise_list = self.start_color_noise_maker()
        else:
            self.color_noise_thread.join(0)
            if not self.color_noise_thread.is_alive():
                self.color_noise = self.color_noise_list[0]
                self.color_noise_list = self.start_color_noise_maker()

        self.data[self.d_idx]['data'] = self.data[self.d_idx]['data'].reshape((ncases*3, cropped_size))
        self.color_noise = self.color_noise[:ncases,:].reshape((3*ncases, 1))
        self.data[self.d_idx]['data'] += self.color_noise * self.color_noise_coeff
        self.data[self.d_idx]['data'] = self.data[self.d_idx]['data'].reshape((ncases, 3* cropped_size))
        self.data[self.d_idx]['data'] *= 1.0 / (1.0 + self.color_noise_coeff) # <--- NOTE: This is the slow line, 0.25sec. Down from 0.75sec when I used division.
    
    def get_next_batch(self):
        self.d_idx = self.batches_generated % 2
        epoch, batchnum = self.curr_epoch, self.curr_batchnum

        self.get_data_from_loader()

        # Subtract mean
        self.data[self.d_idx]['data'] -= self.data_mean_crop
        
        if self.color_noise_coeff > 0 and not self.test:
            self.add_color_noise()
        self.batches_generated += 1
        
        return epoch, batchnum, [self.data[self.d_idx]['data'].T, self.data[self.d_idx]['labvec'].T, self.data[self.d_idx]['labmat'].T]
        
        
    # Takes as input an array returned by get_next_batch
    # Returns a (numCases, imgSize, imgSize, 3) array which can be
    # fed to pylab for plotting.
    # This is used by shownet.py to plot test case predictions.
    def get_plottable_data(self, data, add_mean=True):
        mean = self.data_mean_crop.reshape((data.shape[0],1)) if data.flags.f_contiguous or self.scalar_mean else self.data_mean_crop.reshape((data.shape[0],1))
        return n.require((data + (mean if add_mean else 0)).T.reshape(data.shape[1], 3, self.inner_size, self.inner_size).swapaxes(1,3).swapaxes(1,2) / 255.0, dtype=n.single)
       
class CIFARDataProvider(LabeledDataProvider):
    def __init__(self, data_dir, batch_range=None, init_epoch=1, init_batchnum=None, dp_params=None, test=False):
        LabeledDataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        self.img_size = 32 
        self.num_colors = 3
        self.inner_size =  dp_params['inner_size'] if dp_params['inner_size'] > 0 else self.batch_meta['img_size']
        self.border_size = (self.img_size - self.inner_size) / 2
        self.multiview = dp_params['multiview_test'] and test
        self.num_views = 9
        self.scalar_mean = dp_params['scalar_mean'] 
        self.data_mult = self.num_views if self.multiview else 1
        self.data_dic = []
        for i in batch_range:
            self.data_dic += [unpickle(self.get_data_file_name(i))]
            self.data_dic[-1]["labels"] = n.require(self.data_dic[-1]['labels'], dtype=n.single)
            self.data_dic[-1]["labels"] = n.require(n.tile(self.data_dic[-1]["labels"].reshape((1, n.prod(self.data_dic[-1]["labels"].shape))), (1, self.data_mult)), requirements='C')
            self.data_dic[-1]['data'] = n.require(self.data_dic[-1]['data'] - self.scalar_mean, dtype=n.single, requirements='C')
        
        self.cropped_data = [n.zeros((self.get_data_dims(), self.data_dic[0]['data'].shape[1]*self.data_mult), dtype=n.single) for x in xrange(2)]

        self.batches_generated = 0
        self.data_mean = self.batch_meta['data_mean'].reshape((self.num_colors,self.img_size,self.img_size))[:,self.border_size:self.border_size+self.inner_size,self.border_size:self.border_size+self.inner_size].reshape((self.get_data_dims(), 1))

    def get_next_batch(self):
        epoch, batchnum = self.curr_epoch, self.curr_batchnum
        self.advance_batch()
        bidx = batchnum - self.batch_range[0]

        cropped = self.cropped_data[self.batches_generated % 2]

        self.__trim_borders(self.data_dic[bidx]['data'], cropped)
        cropped -= self.data_mean
        self.batches_generated += 1
        return epoch, batchnum, [cropped, self.data_dic[bidx]['labels']]
        
    def get_data_dims(self, idx=0):
        return self.inner_size**2 * self.num_colors if idx == 0 else 1

    # Takes as input an array returned by get_next_batch
    # Returns a (numCases, imgSize, imgSize, 3) array which can be
    # fed to pylab for plotting.
    # This is used by shownet.py to plot test case predictions.
    def get_plottable_data(self, data):
        return n.require((data + self.data_mean).T.reshape(data.shape[1], 3, self.inner_size, self.inner_size).swapaxes(1,3).swapaxes(1,2) / 255.0, dtype=n.single)
    
    def __trim_borders(self, x, target):
        y = x.reshape(self.num_colors, self.img_size, self.img_size, x.shape[1])

        if self.test: # don't need to loop over cases
            if self.multiview:
                start_positions = [(0,0), (0, self.border_size), (0, self.border_size*2),
                                  (self.border_size, 0), (self.border_size, self.border_size), (self.border_size, self.border_size*2),
                                  (self.border_size*2, 0), (self.border_size*2, self.border_size), (self.border_size*2, self.border_size*2)]
                end_positions = [(sy+self.inner_size, sx+self.inner_size) for (sy,sx) in start_positions]
                for i in xrange(self.num_views):
                    target[:,i * x.shape[1]:(i+1)* x.shape[1]] = y[:,start_positions[i][0]:end_positions[i][0],start_positions[i][1]:end_positions[i][1],:].reshape((self.get_data_dims(),x.shape[1]))
            else:
                pic = y[:,self.border_size:self.border_size+self.inner_size,self.border_size:self.border_size+self.inner_size, :] # just take the center for now
                target[:,:] = pic.reshape((self.get_data_dims(), x.shape[1]))
        else:
            for c in xrange(x.shape[1]): # loop over cases
                startY, startX = nr.randint(0,self.border_size*2 + 1), nr.randint(0,self.border_size*2 + 1)
                endY, endX = startY + self.inner_size, startX + self.inner_size
                pic = y[:,startY:endY,startX:endX, c]
                if nr.randint(2) == 0: # also flip the image with 50% probability
                    pic = pic[:,:,::-1]
                target[:,c] = pic.reshape((self.get_data_dims(),))

class DummyConvNetLogRegDataProvider(LabeledDummyDataProvider):
    def __init__(self, data_dim):
        LabeledDummyDataProvider.__init__(self, data_dim)

        self.img_size = int(sqrt(data_dim/3))
        
    def get_next_batch(self):
        epoch, batchnum, dic = LabeledDummyDataProvider.get_next_batch(self)
        dic = {'data': dic[0], 'labels': dic[1]}
        print dic['data'].shape, dic['labels'].shape
        return epoch, batchnum, [dic['data'], dic['labels']]
    
    # Returns the dimensionality of the two data matrices returned by get_next_batch
    def get_data_dims(self, idx=0):
        return self.batch_meta['num_vis'] if idx == 0 else 1
        
        
#+++++++++++++++++++++++
class CroppedGeneralDataProvider(DLDataProvider2):
    """
    This is the data provider that should generally be used. Several parameters should be passed in a dictionary called
    dp_params when calling ConvNet:
    dataset name: Tuple of moule and dataset name to use
    meta_attribute: which column in the meta field to use
    perm_type: 'random' or otherwise, specifies how to order data before batch creation
    preproc: preprocessing spec which is a dictionary that speciies how to preprocess images (see dldata.stimulus_sets.
    dataset_templates.ImageLoaderPreprocesser for details)

    Here is an example:
    dp_params = {'dataset_name': ('dldata.stimulus_sets.synthetic.synthetic_datasets', 'TrainingDataset'),
    'batch_size': 128,
    'meta_attribute': 'obj',
    'perm_type': 'random',
    'perm_seed': 0,
    'preproc': {'resize_to': (128, 128, 3),
    'mode': 'RGB',
    'dtype': 'float32',
    'normalize': False}
    }
    """
    def __init__(self, data_dir,
            batch_range=None,
            init_epoch=1, init_batchnum=None, dp_params=None, test=False):

        DLDataProvider2.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test, cache_type='hdf5', read_mode='r')

        preproc = dp_params['preproc'] if hasattr(dp_params['preproc'], 'keys') else dp_params['preproc'][0]
        self.num_colors = 1 if preproc['mode'] in ['L', 'L_alpha'] else 3
        self.img_size = preproc['resize_to'][0]

        self.noise_level = dp_params.get('noise_level')
        if not hasattr(dp_params['crop_border'], '__iter__'):
            cb = dp_params['crop_border'] 
        else:
            cb = dp_params['crop_border'][0]
        self.border_size = cb
        self.inner_size = self.img_size - self.border_size*2
        self.multiview = dp_params['multiview_test'] and test

        self.img_flip = dp_params['img_flip']
        if self.img_flip:
            self.num_views = 5*2
        else :
            self.num_views = 5;
        self.data_mult = self.num_views if self.multiview else 1
        print self.border_size
        print self.num_colors
        print self.img_size
        print self.inner_size
        self.batches_generated = 0
        self.data_mean = self.batch_meta['data_mean'].reshape((self.num_colors, self.img_size,
                         self.img_size))[:,self.border_size: self.border_size+self.inner_size,
                                        self.border_size: self.border_size+self.inner_size].reshape((self.get_data_dims(), 1))

    def get_num_views(self):
        return self.num_views

    def get_next_batch(self):
        t0 = time()
        epoch, batchnum, datadic = DLDataProvider2.get_next_batch(self)
        t1 = time()
        if hasattr(datadic['labels'], 'keys'):
            for k in datadic['labels']:
                datadic['labels'][k] = n.require(n.tile(datadic['labels'][k].reshape((1,
                                                    datadic['data'].shape[1])),
                                                    (1, self.data_mult)),
                                      requirements='C')
        else:
            datadic['labels'] = n.require(n.tile(datadic['labels'].reshape((1,
                                                    datadic['data'].shape[1])),
                                                    (1, self.data_mult)),
                                      requirements='C')
        t2 = time()
        # correct for cropped_data size
        cropped = n.zeros((self.get_data_dims(),
                     datadic['data'].shape[1]*self.data_mult), dtype=n.single)
        t3 = time()
        self.__trim_borders(datadic['data'], cropped)
        t4 = time()
        cropped -= self.data_mean
        t5 = time()
        self.batches_generated += 1
        #assert( cropped.shape[1] == datadic['labels'].shape[1] )
        #print('convnet gnb times', t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4)
        
        if hasattr(datadic['labels'], 'keys'):
            bdata = [cropped] + datadic['labels'].values()
        else:
            bdata = [cropped, datadic['labels']]
        
        return epoch, batchnum, bdata

    def get_data_dims(self, idx=0):
        return self.inner_size**2 * self.num_colors if idx == 0 else 1

    def get_out_img_size( self ):
        return self.inner_size

    def get_out_img_depth( self ):
        return self.num_colors

    # Takes as input an array returned by get_next_batch
    # Returns a (numCases, imgSize, imgSize, 3) array which can be
    # fed to pylab for plotting.
    # This is used by shownet.py to plot test case predictions.
    def get_plottable_data(self, data):
        return n.require((data + self.data_mean).T.reshape(data.shape[1], 3, self.inner_size, self.inner_size).swapaxes(1,3).swapaxes(1,2) / 255.0, dtype=n.single)

    def __trim_borders(self, x, target):
        #y = x.reshape(3, 32, 32, x.shape[1])
        if (not self.multiview) and (self.border_size == 0) and (not self.img_flip) and (not self.noise_level):
            t0 = time()
            target[:] = x.copy()
            t1 = time()
            #print('copying time', t1 - t0)
            return 

        y = x.reshape(self.num_colors, self.img_size, self.img_size, x.shape[1])

        if self.test: # don't need to loop over cases
            if self.multiview:
                start_positions = [(0,0),  (0, self.border_size*2),
                                   (self.border_size, self.border_size),
                                  (self.border_size*2, 0), (self.border_size*2, self.border_size*2)]
                end_positions = [(sy+self.inner_size, sx+self.inner_size) for (sy,sx) in start_positions]

                if self.img_flip: # flip image
                    for i in xrange(self.num_views/2):
                        pic = y[:,start_positions[i][0]:end_positions[i][0],start_positions[i][1]:end_positions[i][1],:]
                        target[:,i * x.shape[1]:(i+1)* x.shape[1]] = pic.reshape((self.get_data_dims(),x.shape[1]))
                        target[:,(self.num_views/2 + i) * x.shape[1]:(self.num_views/2 +i+1)* x.shape[1]] = pic[:,:,::-1,:].reshape((self.get_data_dims(),x.shape[1]))

                else :
                    for i in xrange(self.num_views):
                        pic = y[:,start_positions[i][0]:end_positions[i][0],start_positions[i][1]:end_positions[i][1],:]
                        target[:,i * x.shape[1]:(i+1)* x.shape[1]] = pic.reshape((self.get_data_dims(),x.shape[1]))

            else:
                pic = y[:,self.border_size:self.border_size+self.inner_size,self.border_size:self.border_size+self.inner_size, :] # just take the center for now
                target[:,:] = pic.reshape((self.get_data_dims(), x.shape[1]))
        else:
            timer1 = 0
            timer2 = 0
            timer3 = 0
            timer4 = 0
            for c in xrange(x.shape[1]): # loop over cases
                t0 = time()
                startY, startX = nr.randint(0,self.border_size*2 + 1), nr.randint(0,self.border_size*2 + 1)
                endY, endX = startY + self.inner_size, startX + self.inner_size
                t1 = time()
                pic = y[:,startY:endY,startX:endX, c]
                t2 = time()
                if self.img_flip and nr.randint(2) == 0: # also flip the image with 50% probability
                    pic = pic[:,:,::-1]
                if self.noise_level and nr.randint(2) == 0:
                    noise_coefficient = n.random.uniform(low=0, high=self.noise_level)
                    ns = (n.random.RandomState().randn(*pic.shape) * noise_coefficient).astype(pic.dtype)
                    mx = n.abs(pic).max()
                    pic += ns
                    pic *= mx / n.abs(pic).max()
                t3 = time()
                target[:,c] = pic.reshape((self.get_data_dims(),))
                t4 = time()
                timer1 += (t1 - t0)
                timer2 += (t2 - t1)
                timer3 += (t3 - t2)
                timer4 += (t4 - t3)
            print('inner loop timing', timer1, timer2, timer3, timer4)


class CroppedGeneralDataRandomProvider( CroppedGeneralDataProvider ):
    def __init__(self, data_dir,
            img_size, num_colors,  # options i've add to cifar data provider
            batch_range=None,
            init_epoch=1, init_batchnum=None, dp_params=None, test=False):
       CroppedGeneralDataProvider.__init__( self, data_dir,
               img_size, num_colors,
               batch_range,
               init_epoch, init_batchnum, dp_params, test )

    def get_next_batch(self):
        epoch,batchnum, datadic = CroppedGeneralDataProvider.get_next_batch(self)
        # shuffle only training data,never do on testing
        if self.test and self.multiview:
           pass
        else:
            # random shuffle datadic['data'] and datadic['labels']
            num_data = datadic[0].shape[1]
            index = range( num_data )
            r.shuffle(index)
            datadic[0] = n.require( datadic[0][:, index], dtype=n.single, requirements='C' )
            datadic[1] = n.require( datadic[1][:, index], dtype=n.single, requirements='C' )
        return epoch, batchnum, [datadic[0], datadic[1]]

 
class CroppedGeneralDataMapProvider(DLDataMapProvider):
    """
        cropped version of DLDataMapProvider
        takes same arguments as DlDataMapProvider, except dp_params must have a
        key called "crop_border".   dp_params['crop_border'] is either an integer
        or a list of integers of the same length as dp_params["map_methods"] 
        (see doc. for DLDataMapProvider). 
        
        The data_list output for each batch is a list in which 
            -- the 0th element is the "image" map, corresponding to the first map_method output
            -- the 1st element is a label array
            -- the remaining elements are the remaining map_method outputs
        
    """
    def __init__(self, data_dir,
            batch_range=None,
            init_epoch=1, init_batchnum=None, dp_params=None, test=False,
            cache_type='hdf5', read_mode='r'):
            
        DLDataMapProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test, cache_type=cache_type, read_mode=read_mode)

        if hasattr(self.metacol, 'keys'):
            self.num_meta_attrs = len(self.metacol)
        else:
            self.num_meta_attrs = 1

        self.batches_generated = 0
        
        if hasattr(dp_params['crop_border'], '__iter__'):
            border_size_list = dp_params['crop_border']
        else:
            assert isinstance(dp_params['crop_border'], int)
            border_size_list = [dp_params['crop_border']] *  len(self.map_methods)
        
        self.border_size_list = border_size_list
        self.num_colors_list = []
        self.img_size_list = []
        self.inner_size_list = []
        self.data_mean_list = []
        self.multiview = dp_params['multiview_test'] and test
        self.img_flip = dp_params['img_flip']
                    
        for border_size, mname, pp, mshape in zip(border_size_list, self.mnames, self.map_preprocs, self.map_shapes):
            num_colors = mshape[3] if len(mshape) == 4 else 1
            self.num_colors_list.append(num_colors)
            
            img_size = pp['resize_to'][0]
            self.img_size_list.append(img_size)

            inner_size = img_size - 2 * border_size
            self.inner_size_list.append(inner_size) 

            if self.img_flip:
                self.num_views = 5*2
            else :
                self.num_views = 5;
            self.data_mult = self.num_views if self.multiview else 1
            
            data_mean = self.batch_meta_dict[mname]['data_mean']
            nshp0 = (num_colors, img_size, img_size)
            nshp1 = (num_colors * (inner_size ** 2), 1)
            data_mean = data_mean.reshape(nshp0)[:, 
                            border_size: border_size + inner_size,
                            border_size: border_size + inner_size]
            data_mean = data_mean.reshape(nshp1)
            self.data_mean_list.append(data_mean)

    def get_num_views(self):
        return self.num_views

    def get_next_batch(self):
        epoch, batchnum, datadic = DLDataMapProvider.get_next_batch(self)
        map_names = self.mnames
        if hasattr(datadic['labels'], 'keys'):
            for k in datadic['labels']:
                datadic['labels'][k] = n.require(n.tile(datadic['labels'][k].reshape((1,
                                                    datadic[map_names[0]].shape[1])),
                                                    (1, self.data_mult)),
                                                 requirements='C')
        else:
            datadic['labels'] = n.require(n.tile(datadic['labels'].reshape((1,
                                                    datadic[map_names[0]].shape[1])),
                                                    (1, self.data_mult)),
                                          requirements='C')
                                      
        crop_seed = epoch * self.num_batches + batchnum
        for idx, mname in enumerate(map_names):
            ddims = self.get_data_dims(idx if idx == 0 else idx + self.num_meta_attrs)
            cropped = n.zeros((ddims,
                               datadic[mname].shape[1]*self.data_mult), 
                               dtype=n.single)

            border_size = self.border_size_list[idx]
            num_colors = self.num_colors_list[idx]
            img_size = self.img_size_list[idx]
            inner_size = self.inner_size_list[idx]
            self.__trim_borders(datadic[mname], 
                                cropped,
                                border_size,
                                num_colors,
                                img_size,
                                inner_size, 
                                ddims,
                                seed=crop_seed)
            cropped -= self.data_mean_list[idx]
            datadic[mname] = cropped

        data_list = [datadic[map_names[0]]] + \
                    (datadic['labels'].values() if hasattr(datadic['labels'], 'keys') else [datadic['labels']]) + \
                    [datadic[mn] for mn in map_names[1:]]

        self.batches_generated += 1
        return epoch, batchnum, data_list

    def get_data_dims(self, idx=0):
        if (1 <= idx < self.num_meta_attrs + 1):
            return 1
        else:
            if idx >= 1 + self.num_meta_attrs:
                idx = idx - self.num_meta_attrs
            return self.inner_size_list[idx]**2 * self.num_colors_list[idx] 

    def get_out_img_size( self ): 
        return self.inner_size_list[0]**2 

    def get_out_img_depth( self ):
        return self.num_colors_list[0]**2 

    def __trim_borders(self, 
                       x,
                       target,
                       border_size,
                       num_colors,
                       img_size,
                       inner_size,
                       ddims,
                       seed=0):
                       
        if (not self.multiview) and (border_size == 0) and (not self.img_flip):
            target[:] = x.copy()
            return 
            
        y = x.reshape(num_colors, img_size, img_size, x.shape[1])
        
        if self.test: # don't need to loop over cases
            if self.multiview:
                start_positions = [(0,0),  
                                   (0, 2 * self.border_size),
                                   (border_size, border_size),
                                   (2 * border_size, 0), 
                                   (2 * border_size, 2 * self.border_size)]
                end_positions = [(sy + inner_size, sx + inner_size) for (sy, sx) in start_positions]

                if self.img_flip: # flip image
                    for i in xrange(self.num_views/2):
                        pic = y[:,start_positions[i][0]:end_positions[i][0],start_positions[i][1]:end_positions[i][1],:]
                        target[:,i * x.shape[1]:(i+1)* x.shape[1]] = pic.reshape((ddims, x.shape[1]))
                        target[:,(self.num_views/2 + i) * x.shape[1]:(self.num_views/2 +i+1)* x.shape[1]] = pic[:,:,::-1,:].reshape((ddims, x.shape[1]))

                else :
                    for i in xrange(self.num_views):
                        pic = y[:,start_positions[i][0]:end_positions[i][0],start_positions[i][1]:end_positions[i][1],:]
                        target[:,i * x.shape[1]:(i+1)* x.shape[1]] = pic.reshape((ddims, x.shape[1]))

            else:
                pic = y[:, border_size: border_size + inner_size, border_size: border_size + inner_size, :] # just take the center for now
                target[:,:] = pic.reshape((ddims, x.shape[1]))
        else:
            rng = n.random.RandomState(seed=seed)
            for c in xrange(x.shape[1]): # loop over cases
                t0 = time()
                startY, startX = nr.randint(0, 2 * border_size + 1), nr.randint(0, 2 * border_size + 1)
                endY, endX = startY + inner_size, startX + inner_size
                pic = y[:, startY: endY, startX: endX, c]
                if self.img_flip and rng.randint(2) == 0: # also flip the image with 50% probability
                    pic = pic[:,:,::-1]
                target[:,c] = pic.reshape((ddims,))
                

class CroppedImageAndVectorProvider():
    pass



class CroppedGeneralDataProvider1(DLDataProvider):
    """
    This is the data provider that should generally be used. Several parameters should be passed in a dictionary called
    dp_params when calling ConvNet:
    dataset name: Tuple of moule and dataset name to use
    meta_attribute: which column in the meta field to use
    perm_type: 'random' or otherwise, specifies how to order data before batch creation
    preproc: preprocessing spec which is a dictionary that speciies how to preprocess images (see dldata.stimulus_sets.
    dataset_templates.ImageLoaderPreprocesser for details)

    Here is an example:
    dp_params = {'dataset_name': ('dldata.stimulus_sets.synthetic.synthetic_datasets', 'TrainingDataset'),
    'batch_size': 128,
    'meta_attribute': 'obj',
    'perm_type': 'random',
    'perm_seed': 0,
    'preproc': {'resize_to': (128, 128, 3),
    'mode': 'RGB',
    'dtype': 'float32',
    'normalize': False}
    }
    """
    def __init__(self, data_dir,
            batch_range=None,
            init_epoch=1, init_batchnum=None, dp_params=None, test=False):

        DLDataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)

        preproc = dp_params['preproc'] if hasattr(dp_params['preproc'], 'keys') else dp_params['preproc'][0]
        self.num_colors = 1 if preproc['mode'] in ['L', 'L_alpha'] else 3
        self.img_size = preproc['resize_to'][0]

        self.noise_level = dp_params.get('noise_level')
        if not hasattr(dp_params['crop_border'], '__iter__'):
            cb = dp_params['crop_border'] 
        else:
            cb = dp_params['crop_border'][0]
        self.border_size = cb
        self.inner_size = self.img_size - self.border_size*2
        self.multiview = dp_params['multiview_test'] and test

        self.img_flip = dp_params['img_flip']
        if self.img_flip:
            self.num_views = 5*2
        else :
            self.num_views = 5;
        self.data_mult = self.num_views if self.multiview else 1
        print self.border_size
        print self.num_colors
        print self.img_size
        print self.inner_size
        self.batches_generated = 0
        self.data_mean = self.batch_meta['data_mean'].reshape((self.num_colors, self.img_size,
                         self.img_size))[:,self.border_size: self.border_size+self.inner_size,
                                        self.border_size: self.border_size+self.inner_size].reshape((self.get_data_dims(), 1))

    def get_num_views(self):
        return self.num_views

    def get_next_batch(self):
        t0 = time()
        epoch, batchnum, datadic = DLDataProvider.get_next_batch(self)
        t1 = time()
        if hasattr(datadic['labels'], 'keys'):
            for k in datadic['labels']:
                datadic['labels'][k] = n.require(n.tile(datadic['labels'][k].reshape((1,
                                                    datadic['data'].shape[1])),
                                                    (1, self.data_mult)),
                                      requirements='C')
        else:
            datadic['labels'] = n.require(n.tile(datadic['labels'].reshape((1,
                                                    datadic['data'].shape[1])),
                                                    (1, self.data_mult)),
                                      requirements='C')
        t2 = time()
        # correct for cropped_data size
        cropped = n.zeros((self.get_data_dims(),
                     datadic['data'].shape[1]*self.data_mult), dtype=n.single)
        t3 = time()
        self.__trim_borders(datadic['data'], cropped)
        t4 = time()
        cropped -= self.data_mean
        t5 = time()
        self.batches_generated += 1
        #assert( cropped.shape[1] == datadic['labels'].shape[1] )
        #print('convnet gnb times', t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4)
        
        if hasattr(datadic['labels'], 'keys'):
            bdata = [cropped] + datadic['labels'].values()
        else:
            bdata = [cropped, datadic['labels']]
        
        return epoch, batchnum, bdata

    def get_data_dims(self, idx=0):
        return self.inner_size**2 * self.num_colors if idx == 0 else 1

    def get_out_img_size( self ):
        return self.inner_size

    def get_out_img_depth( self ):
        return self.num_colors

    # Takes as input an array returned by get_next_batch
    # Returns a (numCases, imgSize, imgSize, 3) array which can be
    # fed to pylab for plotting.
    # This is used by shownet.py to plot test case predictions.
    def get_plottable_data(self, data):
        return n.require((data + self.data_mean).T.reshape(data.shape[1], 3, self.inner_size, self.inner_size).swapaxes(1,3).swapaxes(1,2) / 255.0, dtype=n.single)

    def __trim_borders(self, x, target):
        #y = x.reshape(3, 32, 32, x.shape[1])
        if (not self.multiview) and (self.border_size == 0) and (not self.img_flip):
            t0 = time()
            target[:] = x.copy()
            t1 = time()
            print('copying time', t1 - t0)
            return 

        y = x.reshape(self.num_colors, self.img_size, self.img_size, x.shape[1])

        if self.test: # don't need to loop over cases
            if self.multiview:
                start_positions = [(0,0),  (0, self.border_size*2),
                                   (self.border_size, self.border_size),
                                  (self.border_size*2, 0), (self.border_size*2, self.border_size*2)]
                end_positions = [(sy+self.inner_size, sx+self.inner_size) for (sy,sx) in start_positions]

                if self.img_flip: # flip image
                    for i in xrange(self.num_views/2):
                        pic = y[:,start_positions[i][0]:end_positions[i][0],start_positions[i][1]:end_positions[i][1],:]
                        target[:,i * x.shape[1]:(i+1)* x.shape[1]] = pic.reshape((self.get_data_dims(),x.shape[1]))
                        target[:,(self.num_views/2 + i) * x.shape[1]:(self.num_views/2 +i+1)* x.shape[1]] = pic[:,:,::-1,:].reshape((self.get_data_dims(),x.shape[1]))

                else :
                    for i in xrange(self.num_views):
                        pic = y[:,start_positions[i][0]:end_positions[i][0],start_positions[i][1]:end_positions[i][1],:]
                        target[:,i * x.shape[1]:(i+1)* x.shape[1]] = pic.reshape((self.get_data_dims(),x.shape[1]))

            else:
                pic = y[:,self.border_size:self.border_size+self.inner_size,self.border_size:self.border_size+self.inner_size, :] # just take the center for now
                target[:,:] = pic.reshape((self.get_data_dims(), x.shape[1]))
        else:
            timer1 = 0
            timer2 = 0
            timer3 = 0
            timer4 = 0
            for c in xrange(x.shape[1]): # loop over cases
                t0 = time()
                startY, startX = nr.randint(0,self.border_size*2 + 1), nr.randint(0,self.border_size*2 + 1)
                endY, endX = startY + self.inner_size, startX + self.inner_size
                t1 = time()
                pic = y[:,startY:endY,startX:endX, c]
                t2 = time()
                if self.img_flip and nr.randint(2) == 0: # also flip the image with 50% probability
                    pic = pic[:,:,::-1]
                if self.noise_level and nr.randint(2) == 0:
                    noise_coefficient = n.random.uniform(low=0, high=self.noise_level)
                    ns = (n.random.RandomState().randn(*pic.shape) * noise_coefficient).astype(pic.dtype)
                    mx = n.abs(pic).max()
                    pic += ns
                    pic *= mx / n.abs(pic).max()
                t3 = time()
                target[:,c] = pic.reshape((self.get_data_dims(),))
                t4 = time()
                timer1 += (t1 - t0)
                timer2 += (t2 - t1)
                timer3 += (t3 - t2)
                timer4 += (t4 - t3)
            print('inner loop timing', timer1, timer2, timer3, timer4)
