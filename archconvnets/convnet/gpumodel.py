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

import datetime
import numpy as n
import os
from time import time, asctime, localtime, strftime
from numpy.random import randn, rand
from numpy import s_, dot, tile, zeros, ones, zeros_like, array, ones_like
from util import *
from data import *
from math import ceil, floor, sqrt
from data import DataProvider, dp_types
import sys
import shutil
import platform
import copy

from options import *


from os import linesep as NL

import pymongo as pm
import gridfs
from yamutils.mongo import SONify


def get_checkpoint_fs(host, port, db_name, fs_name):
    try:
        checkpoint_db = pm.Connection(host=host, port=port)[db_name]
    except pm.errors.ConnectionFailure, e:
        raise(pm.errors.ConnectionFailure(e))
    else:
        return gridfs.GridFS(checkpoint_db, fs_name)


class ModelStateException(Exception):
    pass

# GPU Model interface
class IGPUModel:
    def __init__(self, model_name, op, load_dic, filename_options=None, dp_params={}):
        # these are input parameters
        self.model_name = model_name
        self.op = op
        self.options = op.options
        self.load_dic = load_dic
        self.filename_options = filename_options
        self.get_gpus()
        self.fill_excused_options()
        #assert self.op.all_values_given()

        for o in op.get_options_list():
            setattr(self, o.name, o.value)

        self.dp_params = dp_params
        # these are things that the model must remember but they're not input parameters
        if load_dic:
            self.model_state = load_dic["model_state"]
            if not "experiment_data" in self.options or not self.options["experiment_data"].value_given:
                self.experiment_data = load_dic["rec"]["experiment_data"]
        else:
            self.model_state = {}
            self.model_state["train_outputs"] = []
            self.model_state["test_outputs"] = []
            self.model_state["epoch"] = 1
            self.model_state["batchnum"] = self.train_batch_range[0]
            if not self.options["experiment_data"].value_given:
                idval = model_name + "_" + '_'.join(['%s_%s' % (char, self.options[opt].get_str_value()) for opt, char in filename_options]) + '_' + strftime('%Y-%m-%d_%H.%M.%S')
                self.experiment_data = {'experiment_id': idval}

        self.init_data_providers()
        if load_dic:
            self.train_data_provider.advance_batch()

        # model state often requries knowledge of data provider, so it's initialized after
        try:
            self.init_model_state()
        except ModelStateException, e:
            print e
            sys.exit(1)
        for var, val in self.model_state.iteritems():
            setattr(self, var, val)

        self.import_model()
        self.init_model_lib()

    def import_model(self):
        print "========================="
        print "Importing %s C++ module" % ('_' + self.model_name)
        self.libmodel = __import__('_' + self.model_name)

    def fill_excused_options(self):
        pass

    def init_data_providers(self):
        self.dp_params['convnet'] = self
	print self.dp_params
        try:
            self.test_data_provider = DataProvider.get_instance(
                    self.data_path,
                    self.img_size, self.img_channels,  # options i've add to cifar data provider
                    self.test_batch_range,
                    type=self.dp_type, dp_params=self.dp_params, test=True)
            self.train_data_provider = DataProvider.get_instance(
                    self.data_path,
                    self.img_size, self.img_channels,  # options i've add to cifar data provider
                    self.train_batch_range,
                    self.model_state["epoch"], self.model_state["batchnum"],
                    type=self.dp_type, dp_params=self.dp_params, test=False)
        except DataProviderException, e:
            print "Unable to create data provider: %s" % e
            self.print_data_providers()
            sys.exit()

    def init_model_state(self):
        pass

    def init_model_lib(self):
        pass

    def start(self):
        if self.test_only:
            self.test_outputs += [self.get_test_error()]
            self.print_test_results()
            sys.exit(0)
        self.train()

    def scale_learningRate( self, eps ):
        self.libmodel.scaleModelEps( eps );

    def reset_modelMom( self ):
        self.libmodel.resetModelMom( );

    def train(self):
        print "============================================"
        print "learning rate scale     : ", self.scale_rate
        print "Reset Momentum          : ", self.reset_mom
        print "Image Rotation & Scaling: ", self.img_rs
        print "============================================"
        self.scale_learningRate( self.scale_rate )
        if self.reset_mom:
            self.reset_modelMom( )
        print "========================="
        print "Training %s" % self.model_name
        self.op.print_values()
        print "========================="
        self.print_model_state()
        print "Running on CUDA device(s) %s" % ", ".join("%d" % d for d in self.device_ids)
        print "Current time: %s" % asctime(localtime())
        print "========================="
        next_data = self.get_next_batch()

        #for ii in range(100):
        #    plot_col_image( next_data[2][0][:,ii], 24, 24, 3,
        #            "index: "+str(next_data[2][1][0,ii]) )

        if self.adp_drop:
            dropRate = 0.0
            self.set_dropRate( dropRate );

        # define epoch cost
        epoch_cost = 0
        print_epoch_cost = False
        # training loop
        while self.epoch <= self.num_epochs:
            data = next_data
            self.epoch, self.batchnum = data[0], data[1]
            # reset epoch_cost
            if self.batchnum == 1:
                # print if necessary
                if print_epoch_cost:
                    print "epoch_cost: " + str( epoch_cost )
                # reset epoch_cost
                epoch_cost = 0
                print_epoch_cost = True

            self.print_iteration()
            sys.stdout.flush()

            if self.batchnum == 1 and self.adp_drop:
                dropRate = self.adjust_dropRate( dropRate )

            compute_time_py = time()
            self.start_batch(data)

            # load the next batch while the current one is computing
            next_data = self.get_next_batch()

            batch_output = self.finish_batch()
            self.train_outputs += [batch_output]
            epoch_cost += self.print_train_results()

            if self.get_num_batches_done() % self.testing_freq == 0:
                self.sync_with_host()
                self.test_outputs += [self.get_test_error()]
                self.print_test_results()
                self.print_test_status()
                self.conditional_save()

            self.print_train_time(time() - compute_time_py)
        self.cleanup()

    def cleanup(self):
        sys.exit(0)

    def set_dropRate( self, dropRate ):
        print "set drop rate: ", dropRate
        self.libmodel.setDropRate( dropRate );

    def adjust_dropRate( self, dropRate ):
        #self.print_costs(self.train_outputs[-1])
        #costs, num_cases = cost_outputs[0], cost_outputs[1]
        if not self.train_outputs:
            return dropRate
        costs, num_cases = self.train_outputs[-1][0], self.train_outputs[-1][1]
        for errname in costs.keys():
            #costs[errname] = [(v) for v in costs[errname]]
            #if costs[errname][1] < (1-dropRate) and dropRate <= 0.8:
            if costs[errname][1] < (1-dropRate):
                dropRate += 0.1
                self.set_dropRate( dropRate )

        return dropRate

    def sync_with_host(self):
        self.libmodel.syncWithHost()

    def print_model_state(self):
        pass

    def get_num_batches_done(self):
        return len(self.train_batch_range) * (self.epoch - 1) + self.batchnum - self.train_batch_range[0] + 1

    def get_next_batch(self, train=True):
        dp = self.train_data_provider
        if not train:
            dp = self.test_data_provider
        data = self.parse_batch_data(dp.get_next_batch(), train=train)
        w = dp.get_out_img_size()
        h = dp.get_out_img_size()
        d = dp.get_out_img_depth()

        #plot_col_image( data[2][0][:,0], w, h, d, 'data 0 ' )
        if self.img_rs and train:
            assert( w * h * d == data[2][0].shape[0] )
            self.libmodel.preprocess( [data[2][0]], w, h, d, 0.15, 15 )

        # disply data[2][0] first a few images for debugging
        #plot_col_image( data[2][0][:,0], w, h, d, 'data 0 ' )

        return data

    def parse_batch_data(self, batch_data, train=True):
        return batch_data[0], batch_data[1], batch_data[2]['data']

    def start_batch(self, batch_data, train=True):
        self.libmodel.startBatch(batch_data[2], not train)

    def finish_batch(self):
        return self.libmodel.finishBatch()

    def print_iteration(self):
        print "\t%d.%d..." % (self.epoch, self.batchnum),

    def print_train_time(self, compute_time_py):
        print "(%.3f sec)" % (compute_time_py)

    def print_train_results(self):
        batch_error = self.train_outputs[-1][0]
        if not (batch_error > 0 and batch_error < 2e20):
            print "Crazy train error: %.6f" % batch_error
            self.cleanup()

        print "Train error: %.6f " % (batch_error),

    def print_test_results(self):
        batch_error = self.test_outputs[-1][0]
        print "%s\t\tTest error: %.6f" % (NL, batch_error),

    def print_test_status(self):
        status = (len(self.test_outputs) == 1 or self.test_outputs[-1][0] < self.test_outputs[-2][0]) and "ok" or "WORSE"
        print status,

    def conditional_save(self):
        batch_error = self.test_outputs[-1][0]
        if batch_error > 0 and batch_error < self.max_test_err:
            self.save_state()
        else:
            print "\tTest error > %g, not saving." % self.max_test_err,

    def aggregate_test_outputs(self, test_outputs):
        test_error = tuple([sum(t[r] for t in test_outputs) / (1 if self.test_one else len(self.test_batch_range)) for r in range(len(test_outputs[-1]))])
        return test_error

    def get_test_error(self):
        next_data = self.get_next_batch(train=False)
        test_outputs = []
        while True:
            data = next_data
            self.start_batch(data, train=False)
            load_next = not self.test_one and data[1] < self.test_batch_range[-1]
            if load_next: # load next batch
                next_data = self.get_next_batch(train=False)
            test_outputs += [self.finish_batch()]
            if self.test_only: # Print the individual batch results for safety
                print "batch %d: %s" % (data[1], str(test_outputs[-1]))
            if not load_next:
                break
            sys.stdout.flush()

        return self.aggregate_test_outputs(test_outputs)

    def set_var(self, var_name, var_val):
        setattr(self, var_name, var_val)
        self.model_state[var_name] = var_val
        return var_val

    def get_var(self, var_name):
        return self.model_state[var_name]

    def has_var(self, var_name):
        return var_name in self.model_state

    def save_state(self):
        for att in self.model_state:
            if hasattr(self, att):
                self.model_state[att] = getattr(self, att)
        if self.save_db:
            val_dict = get_convenient_mongodb_representation(self)
            checkpoint_fs = get_checkpoint_fs(self.checkpoint_fs_host,
                                              self.checkpoint_fs_port,
                                              self.checkpoint_db_name,
                                              self.checkpoint_fs_name)
            dic = {"model_state": self.model_state,
                       "op": self.op}
            val_dict['saved_filters'] = True
            if (self.saving_freq > 0) and (((self.get_num_batches_done() / self.testing_freq) % self.saving_freq) == 0):
                val_dict['__save_protected__'] = True
            else:
                val_dict['__save_protected__'] = False
            blob = cPickle.dumps(dic, protocol=cPickle.HIGHEST_PROTOCOL)
            idval = checkpoint_fs.put(blob, **val_dict)
            print('Saved (with filters) to id %s' % str(idval))
            to_remove_filters = list(checkpoint_fs._GridFS__files.find({'experiment_data': self.experiment_data, 
                                                'saved_filters': True,
                                                '__save_protected__': False}).sort('timestamp'))
            for trf in to_remove_filters:
                if trf['_id'] != idval:
                    print('Removing filters saved to id %s' % str(trf['_id']))
                    checkpoint_fs.delete(trf['_id'])
                    blob = cPickle.dumps({}, protocol=cPickle.HIGHEST_PROTOCOL)
                    trf['saved_filters'] = False
                    checkpoint_fs.put(blob, **trf)
                                      

    @staticmethod
    def load_checkpoint(load_dir):
        if os.path.isdir(load_dir):
            return unpickle(os.path.join(load_dir, sorted(os.listdir(load_dir), key=alphanum_key)[-1]))
        return unpickle(load_dir)

    @staticmethod
    def load_checkpoint_from_db(query, checkpoint_fs_host, checkpoint_fs_port, checkpoint_db_name, checkpoint_fs_name, only_rec=False):
        checkpoint_fs = get_checkpoint_fs(checkpoint_fs_host,
                                          checkpoint_fs_port,
                                          checkpoint_db_name,
                                          checkpoint_fs_name)
        query['saved_filters'] = True
        rec = checkpoint_fs._GridFS__files.find(query, sort=[('timestamp', -1)])[0]
        if not only_rec:
            load_dic = cPickle.loads(checkpoint_fs.get_last_version(_id=rec['_id']).read())
        else:
            load_dic = {}
        load_dic['rec'] = rec
        return load_dic

    @staticmethod
    def get_options_parser():
        op = OptionsParser()
        op.add_option("f", "load_file", StringOptionParser, "Load file", default="", excuses=OptionsParser.EXCLUDE_ALL)
        op.add_option("train-range", "train_batch_range", RangeOptionParser, "Data batch range: training")
        op.add_option("test-range", "test_batch_range", RangeOptionParser, "Data batch range: testing")
        op.add_option("data-provider", "dp_type", StringOptionParser, "Data provider", default="default")
        op.add_option("test-freq", "testing_freq", IntegerOptionParser, "Testing frequency", default=25)
        op.add_option("epochs", "num_epochs", IntegerOptionParser, "Number of epochs", default=500)
        op.add_option("data-path", "data_path", StringOptionParser, "Data path", default="")
        op.add_option("max-filesize", "max_filesize_mb", IntegerOptionParser, "Maximum save file size (MB)", default=5000)
        op.add_option("max-test-err", "max_test_err", FloatOptionParser, "Maximum test error for saving")
        op.add_option("num-gpus", "num_gpus", IntegerOptionParser, "Number of GPUs", default=1)
        op.add_option("test-only", "test_only", BooleanOptionParser, "Test and quit?", default=0)
        op.add_option("zip-save", "zip_save", BooleanOptionParser, "Compress checkpoints?", default=0)
        op.add_option("test-one", "test_one", BooleanOptionParser, "Test on one batch at a time?", default=1)
        op.add_option("gpu", "gpu", ListOptionParser(IntegerOptionParser), "GPU override", default=OptionExpression("[-1] * num_gpus"))
        ####### my additionial config #########
        op.add_option("scale-rate", "scale_rate", FloatOptionParser, "Learning Rate Scale Factor", default=1 )
        op.add_option("adp-drop", "adp_drop", BooleanOptionParser, "Adaptive Drop Training", default=False )
        op.add_option("reset-mom", "reset_mom", BooleanOptionParser, "Reset layer momentum",
              default=False )
        ####### db configs #######
        op.add_option("save-db", "save_db", BooleanOptionParser, "Save checkpoints to mongo database?", default=0)
        op.add_option("saving-freq", "saving_freq", IntegerOptionParser, 
                      "Frequency for saving filters to db filesystem, as a multiple of testing-freq", 
                      default=1)
        op.add_option("checkpoint-fs-host", "checkpoint_fs_host", StringOptionParser, "Host for Saving Checkpoints to DB", default="localhost")
        op.add_option("checkpoint-fs-port", "checkpoint_fs_port", IntegerOptionParser, "Port for Saving Checkpoints to DB", default=27017)
        op.add_option("checkpoint-db-name", "checkpoint_db_name", StringOptionParser,
                "Name for mongodb database for saved checkpoints", default="convnet_checkpoint_db")
        op.add_option("checkpoint-fs-name", "checkpoint_fs_name", StringOptionParser,
                "Name for gridfs FS for saved checkpoints", default="convnet_checkpoint_fs")
        op.add_option("experiment-data", "experiment_data", JSONOptionParser, "Data for grouping results in database", default="")
        op.add_option("load-query", "load_query", JSONOptionParser, "Query for loading checkpoint from database", default="", excuses=OptionsParser.EXCLUDE_ALL)
        return op

    @staticmethod
    def print_data_providers():
        print "Available data providers:"
        for dp, desc in dp_types.iteritems():
            print "    %s: %s" % (dp, desc)

    def get_gpus(self):
        self.device_ids = [get_gpu_lock(g) for g in self.op.get_value('gpu')]
        if GPU_LOCK_NO_LOCK in self.device_ids:
            print "Not enough free GPUs!"
            sys.exit()

    @staticmethod
    def parse_options(op, input_opts=None, ignore_argv=False):
        try:
            load_dic = None
            options = op.parse(input_opts=input_opts, ignore_argv=ignore_argv)
            if "experiment_data" in options and options["experiment_data"].value_given:
                assert "experiment_id" in options["experiment_data"].value
            if options["load_file"].value_given:
                load_dic = IGPUModel.load_checkpoint(options["load_file"].value)
            if options["load_query"].value_given:
                print "loadin from db"
                load_dic = IGPUModel.load_checkpoint_from_db(options["load_query"].value,
                                                             options["checkpoint_fs_host"].value,
                                                             options["checkpoint_fs_port"].value,
                                                             options["checkpoint_db_name"].value,
                                                             options["checkpoint_fs_name"].value)
            if load_dic is not None:
                old_op = load_dic["op"]
                old_op.merge_from(op)
                op = old_op
            op.eval_expr_defaults()
            return op, load_dic
        except OptionMissingException, e:
            print e
            op.print_usage()
        except OptionException, e:
            print e
        except UnpickleError, e:
            print "Error loading checkpoint:"
            print e
        sys.exit()


def get_convenient_mongodb_representation(self):
    val_dict = get_convenient_mongodb_representation_base(self.op, self.model_state)
    val_dict['epoch'] = self.epoch
    val_dict['batch_num'] = self.batchnum
    val_dict['timestamp'] = datetime.datetime.utcnow()
    val_dict['experiment_data'] = self.experiment_data
    return SONify(val_dict)


def get_convenient_mongodb_representation_base(op, model_state):
    val_dict = dict([(_o.name, _o.value) for _o in op.get_options_list()])
    def make_mongo_safe(_d):
        for _k in _d:
            if '.' in _k:
                _d[_k.replace('.', '___')] = _d.pop(_k)
    if 'load_query' in val_dict:
        val_dict['load_query'] = make_mongo_safe(val_dict['load_query'])

    for k in model_state:
        if k == 'layers':
            bad_keys = ['weights', 'inputLayers', 'biasesInc',
                        'biases', 'weightsInc']
            layers = copy.deepcopy(model_state[k])
            for _l in layers:
                for _bk in bad_keys:
                    if _bk in _l:
                        _l.pop(_bk)
            layers = dict([(_l['name'], _l) for _l in layers])
            val_dict[k] = layers
        elif k == 'train_outputs':
            tfreq = val_dict['testing_freq']
            val_dict[k] = model_state[k][-tfreq:]
        elif k == 'test_outputs':
            val_dict[k] = model_state[k][-1]
        else:
            val_dict[k] = model_state[k]

    return SONify(val_dict)

