# image net data provider

from PIL import Image
from util import pickle,unpickle
import numpy as n
import sys
from numpy.random import random_integers
from time import time, asctime, localtime, strftime
from math import *

MEAN_FILE_EXT = "_mean"

def PIL2array(img):
   #if img.mode == 'L':
   #   r = n.array(img.getdata(), n.uint8).reshape(img.size[1], img.size[0] )
   #   result = n.zeros( (img.size[1], img.size[0],3 ), n.uint8 )
   #   result[:,:,0] = r
   #   result[:,:,1] = r
   #   result[:,:,2] = r
   #   return result
   #else:
   #   return n.array(img.getdata(), n.uint8).reshape(img.size[1], img.size[0], 3)
   if img.mode == 'L':
      I = n.asarray( img )
      result = n.zeros( (img.size[1], img.size[0],3 ), n.uint8 )
      result[:,:,0] = I
      result[:,:,1] = I
      result[:,:,2] = I
      return result
   else:
      return n.asarray( img )

def array2PIL(arr):
   return Image.fromarray( n.uint8(arr) )

class ImagenetDataProvider:
   def __init__( self, data_file, root_path, data_mode = "train", random_transform = False,
           batch_size = 128, crop_width = 224, crop_height = 224 ):
      # read image-name image-index map from file
      self.data = unpickle( data_file )
      self.num_classes = len( self.data['index_map'] )
      self.data_mode = data_mode
      self.random_transform = random_transform
      self.root_path = root_path
      if data_mode == "all":
         index_map = self.data['index_map']
      elif data_mode == "val":
         index_map = self.data['index_map_val']
      elif data_mode == "train":
         index_map = self.data['index_map_train']
      else:
         print "data_mode: " + data_mode + " not valid"
         import pdb; pdb.set_trace()
         sys.exit(1)

      # get batch queue
      self.batch_queue = []
      has_add = True
      while has_add:
         has_add = False
         for i in range( self.num_classes ):
            if len(index_map[i]) > 0:
               index = index_map[i].pop()
               self.batch_queue.append( index )
               has_add = True

      self.num_images = len( self.batch_queue )

      #init current index and batch size
      self.batch_size = batch_size
      self.prev_batch_size = batch_size
      self.crop_width = crop_width
      self.crop_height = crop_height
      self.batch_index = 1
      self.epoch = 1
      
      # read data mean from file
      data_mean_file = unpickle( data_file + MEAN_FILE_EXT )
      self.data_mean = data_mean_file['data']

   def get_data_dims( self, idx ):
      if idx == 0:
         return self.crop_width * self.crop_height * 3
      if idx == 1:
         return 1

   def get_previous_batch_size( self ):
      return self.prev_batch_size

   def get_next_batch( self ):
      # construct next batch online
      # batch_data[0]: epoch
      # batch_data[1]: batchnum
      # batch_data[2]['label']: each column represents an image 
      # batch_data[2]['data'] : each column represents an image 
      # this function only crop center 256 x 256 in image for classification


      total_time_start = time()

      alloc_time_start = time()
      result_data = n.zeros( ( self.crop_width * self.crop_height * 3, self.batch_size ), \
            n.float32 )
      result_label = n.zeros( (1,self.batch_size ), n.float32 )

      batch_index = self.batch_index - 1
      if batch_index * self.batch_size >= self.num_images:
         self.batch_index = 1
         self.epoch += 1
         batch_index = 0
      alloc_time = time() - alloc_time_start

      # loading/tranform image time
      load_time = 0
      transform_time = 0

      lt_time_start = time()
      k = 0
      for i in range( self.batch_size ):
         index = (i + batch_index * self.batch_size ) 
         if index >= self.num_images:
            break
         k += 1
         index = self.batch_queue[index]
         result_data[:,i], result_label[0,i], lti, tti = self.get_data_label( index )
         load_time += lti
         transform_time += tti
      lt_time = time() - lt_time_start 

      pack_time_start = time()
      # shrink result_data, result_label to have k columns
      if k < self.batch_size:
         result_data = result_data[:,0:k]
         result_label = result_label[0,0:k].reshape(1,k)
         self.previous_batch_size = k

      self.batch_index += 1
      result = {}
      result['data'] = result_data
      result['label'] = result_label
      #result['label'] = result_label % 10
      #import pdb; pdb.set_trace()
      pack_time = time() - pack_time_start
      print "load data: (%.3f sec) " % ( time() - total_time_start ),
      print " = %.2f(%.2f + %.2f) + %.2f" % (lt_time, load_time , transform_time, alloc_time), 
      return self.epoch, batch_index+1, result

   def get_data_label( self, index ):
      #import pdb; pdb.set_trace()
      image_path = self.root_path + "/" + self.data['image_path'][index]
      label = self.data['image_label'][index]

      #load image 
      load_time_start= time()
      im = Image.open( image_path )
      image_matrix = PIL2array( im )
      load_time = time() - load_time_start 

      # generate transformed image
      transform_time_start = time()
      #[x,y,w,h] = im.getbbox()
      x = 0
      y = 0
      (w,h) = im.size

      # get image matrix and substract mean
      image_matrix = image_matrix.astype(n.float32)
      image_matrix -= self.data_mean

      if self.random_transform:
          # random crop
          x += random_integers( 0, w - self.crop_width - 1)
          y += random_integers( 0, h - self.crop_height - 1)
      else:
          # fixed crop
          x += (w - self.crop_width)/2
          y += (h - self.crop_height)/2

      #crop image
      assert( x + self.crop_width < w )
      assert( y + self.crop_height < h )
      #im = im.crop( (x,y, x + self.crop_width, y + self.crop_height ) )
      image_matrix = image_matrix[ x:x+self.crop_width, y:y+self.crop_width, : ]

      if self.random_transform:
          # flip: roll a dice to whether flip image
          if random_integers( 0,1 ) > 0.5:
              #im = im.transpose( Image.FLIP_LEFT_RIGHT )
              image_matrix = image_matrix[:, -1::-1, :]

      image_matrix = image_matrix.reshape( (self.crop_width * self.crop_height * 3, ) )
      image_matrix = n.require( image_matrix, dtype=n.single, 
            requirements='C')
      label = n.require( label, dtype=n.single, requirements='C' )

      transform_time = time() - transform_time_start 
      return image_matrix, label, load_time, transform_time;

   def get_num_classes( self ):
      return self.num_classes

   def get_num_batches( self ):
      return  int(ceil( 1.0 * len(self.batch_queue) / self.batch_size ))

   def print_data_summary( self ):
      class_labels = [ self.data['image_label'][x] for x in self.batch_queue ]
      label_hist = [0] * self.get_num_classes() 
      for i in range( len(class_labels ) ):
         label_hist[ class_labels[i] ] += 1
      print "Class Label Hist: ", label_hist, len(label_hist)
      print "Num Batches     : ", self.get_num_batches()

if __name__ == "__main__":
   data_file = '/home/snwiz/data/imagenet12/code/data/imagenet_data_tiny10'
   provider = ImagenetDataProvider( data_file, 'val', batch_size = 128, random_transform = True )
   for i in range(2000):
      epoch, batch_index, data = provider.get_next_batch()
      print 'epoch: ' + str(epoch) + ' batch_index: ' + str(batch_index) + \
            '/' + str(provider.get_num_batches()) + \
            ' data:  ' + str(data['data'][0:5,0:5]) +\
            ' label: ' + str(data['label'][0:5,0:5] )
