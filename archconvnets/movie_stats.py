# -*- coding: utf-8 -*-
# <nbformat>2</nbformat>

# <codecell>

from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from scipy.stats import rankdata
import scipy.io as sio
from scipy.io import loadmat
from scipy.io import savemat
from scipy.stats.mstats import zscore
import scipy.io as sio
import numpy as np
import tabular as tb
import copy
from numpy import sqrt
from numpy import mean
from numpy import median
from numpy import std
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import random
from sklearn import preprocessing
import pickle as pk
import time
#import dldata_old.HvM.neural_datasets as nd
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import random
from scipy.stats import rankdata
#from dldata_old.HvM.utils import NKlike
import os
#d = nd.HvMWithDiscfade()

# <codecell>

file = open('/home/darren/test_errs','r')
test_errs = pk.load(file)
file.close()

# generate epoch names
epoch_names = []
epoch_names_test = []
epoch_names_test_inds = []
ind = 0
offset = 100
for epoch in np.arange(1,13):
    offset2 = 0
    offset -= 1
    while True:
        epoch_names.append(str(epoch) + '.' + str(offset + 100*offset2))
        if offset2 == 0:
            epoch_names_test.append(str(epoch) + '.' + str(offset + 100*offset2))
            epoch_names_test_inds.append(ind)
        offset2 += 1
        ind += 1
        if (offset + 100*offset2) > 9100:
            break;
epoch_names = np.asarray(epoch_names)
epoch_names_test = np.asarray(epoch_names_test)
epoch_names_test_inds = np.asarray(epoch_names_test_inds)

nm = 'ConvNet_full_nofc'
#layer = 'pool2_6a'
layer = 'pool3_11a'

n_points = 150
epoch_names_test_inds = np.array(np.round(np.linspace(0,len(epoch_names)-1, n_points)),dtype='int')
epoch_names_test = epoch_names[epoch_names_test_inds]
n_movies = 221
n_frames_per_movie = 150
batch_size = 128

# <codecell>

n_points = len(epoch_names_test)

sq_err_mean = np.zeros((n_points,n_movies))
sq_err_median = np.zeros((n_points,n_movies))
s_corr = np.zeros((n_points,n_movies))
p_corr = np.zeros((n_points,n_movies))
pdist_mean = np.zeros((n_points,n_movies))
pdist_median = np.zeros((n_points,n_movies))

sq_err_mean_mean = np.zeros(n_points)
sq_err_mean_median = np.zeros(n_points)
sq_err_median_mean = np.zeros(n_points)
sq_err_median_median = np.zeros(n_points)
s_corr_mean = np.zeros(n_points)
s_corr_median = np.zeros(n_points)
p_corr_mean = np.zeros(n_points)
p_corr_median = np.zeros(n_points)
pdist_mean_mean = np.zeros(n_points)
pdist_mean_median = np.zeros(n_points)
pdist_median_mean = np.zeros(n_points)
pdist_median_median = np.zeros(n_points)

ind = 0
n_batches = 257

for time_point in epoch_names_test:
    t_start = time.time()
    os.system('rm /tmp/movie_' + layer + '/data*')
    os.system('python /home/darren/archconvnets_write/archconvnets_write/convnet/shownet.py -f /export/imgnet_storage_full/' + nm + '/' + time_point + ' --test-range=20000-' + str(20000 + n_batches) + ' --train-range=0 --write-features=' + layer + ' --feature-path=/tmp/movie_' + layer)
    print time_point, ind, len(epoch_names_test), time.time() - t_start

    file = open('/tmp/movie_' + layer + '/data_batch_20025','r')
    x = pk.load(file)
    n_features = x['data'].shape[1]
    features = np.zeros((n_frames_per_movie, n_features))
    feature_inds = np.arange(n_features)
    imgs_copy = copy.deepcopy(batch_size)
    imgs_copied = 0
    offset = 0
    movie_ind = 0
    for i in np.arange(20000, 20000 + n_batches):
        file = open('/tmp/movie_' + layer + '/data_batch_' + str(i),'r')
        x = pk.load(file)
        features[imgs_copied:imgs_copied+imgs_copy] = copy.deepcopy(x['data'][:imgs_copy])
        imgs_copied += imgs_copy
        if imgs_copied >= n_frames_per_movie:
            features = preprocessing.scale(features, axis=1)
            # finished loading movie, compute statistics
            for frame in np.arange(n_frames_per_movie-1):
                sq_err_mean[ind, movie_ind] += np.mean((features[frame] - features[frame+1])**2)
                sq_err_median[ind, movie_ind] += np.median((features[frame] - features[frame+1])**2)
                p_corr[ind, movie_ind] = pearsonr(features[frame],features[frame+1])[0]
                s_corr[ind, movie_ind] = spearmanr(features[frame],features[frame+1])[0]
            random.shuffle(feature_inds)
            features_sample = copy.deepcopy(features[:,feature_inds[:200]])
            features_sample = np.transpose(features_sample, (1,0))
            features_pdist = pdist(features_sample,'correlation')
            pdist_mean[ind, movie_ind] = np.mean(features_pdist)
            pdist_median[ind, movie_ind] = np.median(features_pdist)
            
            #begin loading next movie
            movie_ind += 1
            offset = copy.deepcopy(imgs_copy)
            imgs_copy = batch_size - imgs_copy
            features[:imgs_copy] = copy.deepcopy(x['data'][offset:offset+imgs_copy])
            imgs_copied = copy.deepcopy(imgs_copy)
            imgs_copy = np.min((n_frames_per_movie - imgs_copy,  batch_size))
        else:
            imgs_copy = np.min((n_frames_per_movie - imgs_copied,  batch_size))
            offset = 0
        file.close()
    sq_err_mean_mean[ind] = np.mean(sq_err_mean[ind])
    sq_err_mean_median[ind] = np.median(sq_err_mean[ind])
    sq_err_median_mean[ind] = np.mean(sq_err_median[ind])
    sq_err_median_median[ind] = np.median(sq_err_median[ind])
    p_corr_mean[ind] = np.mean(p_corr[ind])
    p_corr_median[ind] = np.median(p_corr[ind])
    s_corr_mean[ind] = np.mean(s_corr[ind])
    s_corr_median[ind] = np.median(s_corr[ind])
    pdist_mean_mean[ind] = np.mean(pdist_mean[ind])
    pdist_mean_median[ind] = np.median(pdist_mean[ind])
    pdist_median_mean[ind] = np.mean(pdist_median[ind])
    pdist_median_median[ind] = np.median(pdist_median[ind])
    
    print sq_err_mean_mean[ind], sq_err_mean_median[ind], sq_err_median_mean[ind], sq_err_median_median[ind]
    print p_corr_mean[ind], p_corr_median[ind], s_corr_mean[ind], s_corr_median[ind]
    if ind > 0:
        errs = np.squeeze(test_errs[epoch_names_test_inds[:ind+1]])
        print pearsonr(np.squeeze(sq_err_mean_mean[:ind+1]), errs), spearmanr(np.squeeze(sq_err_mean_mean[:ind+1]), errs)
        print pearsonr(np.squeeze(sq_err_mean_median[:ind+1]), errs), spearmanr(np.squeeze(sq_err_mean_median[:ind+1]), errs)
        print pearsonr(np.squeeze(sq_err_median_mean[:ind+1]), errs), spearmanr(np.squeeze(sq_err_median_mean[:ind+1]), errs)
        print pearsonr(np.squeeze(sq_err_median_median[:ind+1]), errs), spearmanr(np.squeeze(sq_err_median_median[:ind+1]), errs)
        print
        print pearsonr(np.squeeze(p_corr_mean[:ind+1]), errs), spearmanr(np.squeeze(p_corr_mean[:ind+1]), np.squeeze(test_errs[epoch_names_test_inds[:ind+1]]))
        print pearsonr(np.squeeze(p_corr_median[:ind+1]), errs), spearmanr(np.squeeze(p_corr_median[:ind+1]), np.squeeze(test_errs[epoch_names_test_inds[:ind+1]]))
        print pearsonr(np.squeeze(s_corr_mean[:ind+1]), errs), spearmanr(np.squeeze(s_corr_mean[:ind+1]), np.squeeze(test_errs[epoch_names_test_inds[:ind+1]]))
        print pearsonr(np.squeeze(s_corr_median[:ind+1]), errs), spearmanr(np.squeeze(s_corr_median[:ind+1]), np.squeeze(test_errs[epoch_names_test_inds[:ind+1]]))
        print
        print pearsonr(np.squeeze(pdist_mean_mean[:ind+1]), errs), spearmanr(np.squeeze(pdist_mean_mean[:ind+1]), errs)
        print pearsonr(np.squeeze(pdist_mean_median[:ind+1]), errs), spearmanr(np.squeeze(pdist_mean_mean[:ind+1]), errs)
        print pearsonr(np.squeeze(pdist_median_mean[:ind+1]), errs), spearmanr(np.squeeze(pdist_median_mean[:ind+1]), errs)
        print pearsonr(np.squeeze(pdist_median_median[:ind+1]), errs), spearmanr(np.squeeze(pdist_median_median[:ind+1]), errs)
    ind += 1
    savemat('/home/darren/movie_' + layer + '_npoints' + str(n_points) + '.mat', {'sq_err_mean': sq_err_mean,
                                                     'sq_err_median': sq_err_median,
                                                     'sq_err_mean_mean': sq_err_mean_mean,
                                                     'sq_err_mean_median': sq_err_mean_median,
                                                     'sq_err_median_mean': sq_err_median_mean,
                                                     'sq_err_median_median': sq_err_median_median,
                                                     'p_corr_mean': p_corr_mean,
                                                     'p_corr_median': p_corr_median,
                                                     's_corr_mean': s_corr_mean,
                                                     's_corr_median': s_corr_median,
                                                     'pdist_mean_mean': pdist_mean_mean,
                                                     'pdist_mean_median': pdist_mean_median,
                                                     'pdist_median_mean': pdist_median_mean,
                                                     'pdist_median_median': pdist_median_median,
                                                     'test_errs': test_errs,
                                                     'epoch_names_test_inds': epoch_names_test_inds})
    print time.time() - t_start

# <codecell>


