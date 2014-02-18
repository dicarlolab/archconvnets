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
import pickle as pk
import numpy
import sys
import getopt as opt
from util import *
from math import sqrt, ceil, floor
import os
from gpumodel import IGPUModel
import random as r
import numpy.random as nr
from convnet import ConvNet
from options import *
import pymongo as pm
import gridfs
import copy
import collections
from yamutils.mongo import SONify
from pymongo.errors import ConnectionFailure
import datetime

try:
    FEATURE_DB_PORT = int(os.environ.get('FEATURE_DB_PORT', 22334))
    FEATURE_DB = pm.MongoClient('localhost', port=FEATURE_DB_PORT)['features']
except ConnectionFailure:
    print 'Feature database not configured'
    FEATURE_COLLECTION = None


class ExtractNetError(Exception):
    pass


class ExtractConvNet(ConvNet):
    def __init__(self, op, load_dic, dp_params=None):
        """
        NOTE: preprocs are sorted before being stored with every entry when write-db is used
        """
        ConvNet.__init__(self, op, load_dic, dp_params=dp_params)
        self.feature_path = op.get_value('feature_path')
        if op.get_value('write_db'):
            self.coll = self.get_feature_coll(op.get_value('dp_params')['dataset_name'])
            self.coll.ensure_index([('id', 1), ('feature_layer', 1), ('model_id', 1), ('dp_params.preproc', 1)],
                                   unique=True)
            self.ind_set = self.test_data_provider.get_indset()
            try:
                self.meta = self.test_data_provider.meta
            except ValueError:
                raise ExtractNetError('Writing to database only supported using a data provider that has a meta')
            try:
                print self.load_dic['rec']['_id']
                model_id = self.load_dic['rec']['_id']
                dp_params = op.get_value('dp_params')
                preproc = collections.OrderedDict(sorted(dp_params['preproc'].items()))  # Mongo cares about order
                dp_params['preproc'] = preproc
                self.base_record = {'checkpoint_fs_host': op.get_value('checkpoint_fs_host'),
                                    'checkpoint_fs_port': op.get_value('checkpoint_fs_port'),
                                    'checkpoint_db_name': op.get_value('checkpoint_db_name'),
                                    'checkpoint_fs_name': op.get_value('checkpoint_fs_name'),
                                    'feature_layer': op.get_value('feature_layer'),
                                    'dp_params': dp_params,
                                    'model_id': model_id}
            except KeyError:
                raise ExtractNetError('Writing to database only supported using models stored in database')

    def get_feature_coll(self, dataset_name):
        collection_name = str(dataset_name[0]) + str(dataset_name[1])
        coll = FEATURE_DB[collection_name]
        return coll

    def init_model_state(self):
        ConvNet.init_model_state(self)
        self.ftr_layer_idx = self.get_layer_idx(self.op.get_value('feature_layer'))

    def do_write_features(self):
        print self.feature_path
        if not os.path.exists(self.feature_path):
            os.makedirs(self.feature_path)
        next_data = self.get_next_batch(train=False)
        b1 = next_data[1]
        num_ftrs = self.layers[self.ftr_layer_idx]['outputs']
        while True:
            batch = next_data[1]
            data = next_data[2]
            ftrs = n.zeros((data[0].shape[1], num_ftrs), dtype=n.single)
            self.libmodel.startFeatureWriter(data + [ftrs], self.ftr_layer_idx, 1)

            # load the next batch while the current one is computing
            next_data = self.get_next_batch(train=False)
            self.finish_batch()
            if self.op.get_value('write_db'):
                self.write_features_to_db(ftrs, batch, data)
            if self.op.get_value('write_disk'):
                path_out = os.path.join(self.feature_path, 'data_batch_%d' % batch)
                pickle(path_out, {'data': ftrs, 'labels': data[1]})
                print "Wrote feature file %s" % path_out
            if next_data[1] == b1:
                break
        if self.op.get_value('write_disk'):
            pickle(os.path.join(self.feature_path, 'batches.meta'), {'source_model': self.load_file,
                                                                     'source_model_query': self.load_query,
                                                                     'num_vis': num_ftrs})
        if self.op.get_value('write_db'):  # log the extraction
            record = {'dataset_name': op.get_value('dp_params')['dataset_name'],
                      'feature_layer': op.get_value('feature_layer'),
                      'preproc': self.base_record['dp_params']['preproc'],
                      'model_id': self.base_record['model_id'],
                      'extraction_time': datetime.datetime.now().isoformat()}
            extraction_logs = self.get_extraction_log_collection()
            extraction_logs.insert(SONify(record))

    def get_extraction_log_collection(self):
        return FEATURE_DB['logs']

    def write_features_to_db(self, ftrs, batch, data):
        print 'Uploading batch %d to database' % batch
        lists = [list(ftr) for ftr in ftrs]
        inds = self.ind_set[batch]
        ids = self.meta['id'][inds]

        query = {'dp_params.preproc': self.base_record['dp_params']['preproc'],
                 'feature_layer': self.base_record['feature_layer'],
                 'model_id': self.base_record['model_id'],
                 'id': {'$in': ids}}
        self.coll.remove(SONify(query))
        for ftr, img_id in zip(lists, ids):
            record = copy.deepcopy(self.base_record)
            record['id'] = img_id
            record['feature'] = ftr
            self.coll.insert(SONify(record))

    def start(self):
        self.op.print_values()
        self.do_write_features()

    @classmethod
    def get_options_parser(cls):
        op = ConvNet.get_options_parser()
        for option in list(op.options):
            if option not in ('gpu', 'load_file', 'train_batch_range', 'test_batch_range',
                              'load_query', 'checkpoint_fs_host', 'checkpoint_fs_port',
                              'checkpoint_db_name', 'checkpoint_fs_name', 'data_path',
                              'dp_type', 'dp_params', 'img_size'):
                op.delete_option(option)
        op.add_option("feature-layer", 'feature_layer',
                      StringOptionParser, "Write test data features from given layer",
                      default="")
        op.add_option("feature-path", "feature_path",
                      StringOptionParser, "Write test data features to this path (to be used with --layer)",
                      default="")
        op.add_option("write-disk", "write_disk",
                      BooleanOptionParser, "Write test data features from --layer to --feature-path)",
                      default=True)
        op.add_option("write-db", "write_db",
                      BooleanOptionParser, "Write all data features from the dataset to mongodb in standard format",
                      default=False)

        op.options['load_file'].default = None
        return op


import cPickle

if __name__ == "__main__":
    try:
        op = ExtractConvNet.get_options_parser()
        op, load_dic = IGPUModel.parse_options(op)
        model = ExtractConvNet(op, load_dic)
        model.start()
    except (UnpickleError, ExtractNetError, opt.GetoptError), e:
        print "----------------"
        print "Error:"
        print e 
