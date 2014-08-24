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

import numpy as n
import os
from time import time, asctime, localtime, strftime
from util import *
from data import *
from options import *
from math import ceil, floor, sqrt
from data import DataProvider, dp_types
import sys
import shutil
import platform
from os import linesep as NL
from threading import Thread
import tempfile as tf
import datetime

import copy
import collections
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
        
        
def get_recent_checkpoint_fs(host, port, db_name, fs_name):
    return get_checkpoint_fs(host, port, db_name + "__RECENT", fs_name)


def cleanup_checkpoint_db(host, port, db_name, fs_name, keep_recents=True):

    rfs = get_recent_checkpoint_fs(host, port, db_name, fs_name)

    if keep_recents:
        fs = get_checkpoint_fs(host, port, db_name, fs_name)
        rcoll = rfs._GridFS__files
        edatas = rcoll.distinct('experiment_data')
        for ed in edatas:
            rec = rcoll.find({'experiment_data': ed},
                             as_class=collections.OrderedDict).sort([('timestamp', -1)])[0]
            blob = rfs.get_last_version(_id=rec['_id'])
            idval = fs.put(blob, **rec)    
            print('saved record with filters to id %s' % repr(idval))
        
    conn = pm.Connection(host=host, port=port)
    conn.drop_database(rfs._GridFS__database)


class ModelStateException(Exception):
    pass
    

class NoMatchingCheckpointError(Exception):
    pass


class CheckpointWriter(Thread):
    def __init__(self, dic,
                       checkpoint_fs_host,
                       checkpoint_fs_port,
                       checkpoint_db_name,
                       checkpoint_fs_name,
                       save_filters,
                       save_recent_filters,
                       saving_freq,
                       testing_freq,
                       epoch,
                       batchnum,
                       num_batches_done,
                       experiment_data):
        Thread.__init__(self)
        self.dic = dic
        self.checkpoint_fs_host = checkpoint_fs_host
        self.checkpoint_fs_port = checkpoint_fs_port
        self.checkpoint_db_name = checkpoint_db_name
        self.checkpoint_fs_name = checkpoint_fs_name
        self.save_filters = save_filters
        self.save_recent_filters = save_recent_filters 
        self.saving_freq = saving_freq
        self.testing_freq = testing_freq
        self.epoch = epoch
        self.batchnum = batchnum
        self.num_batches_done = num_batches_done
        self.experiment_data = experiment_data
        
    def run(self):
    
        dic = self.dic

        val_dict = get_convenient_mongodb_representation_base(dic['op'], dic['model_state'])
        val_dict['epoch'] = self.epoch
        val_dict['batch_num'] = self.batchnum
        val_dict['timestamp'] = datetime.datetime.utcnow()
        val_dict['experiment_data'] = self.experiment_data
        val_dict = SONify(val_dict)    
    
        checkpoint_fs = get_checkpoint_fs(self.checkpoint_fs_host,
                                          self.checkpoint_fs_port,
                                          self.checkpoint_db_name,
                                          self.checkpoint_fs_name)
        
        
        if self.save_filters and (self.saving_freq > 0) and (((self.num_batches_done / self.testing_freq) % self.saving_freq) == 0):
            val_dict['saved_filters'] = True
            save_dic = dic
            msg = 'Saved (with filters) to id %s'
            save_recent = False
        else:
            val_dict['saved_filters'] = False
            msg = 'Saved (without filters) to id %s'
            save_dic = collections.OrderedDict()
            save_recent = self.save_filters and self.save_recent_filters
        blob = cPickle.dumps(save_dic, protocol=cPickle.HIGHEST_PROTOCOL)
        idval = checkpoint_fs.put(blob, **val_dict)
        print(msg % str(idval))
        
        if save_recent:
            checkpoint_recent_fs = get_recent_checkpoint_fs(self.checkpoint_fs_host,
                                          self.checkpoint_fs_port,
                                          self.checkpoint_db_name,
                                          self.checkpoint_fs_name)
            val_dict['saved_filters'] = True
            blob = cPickle.dumps(dic, protocol=cPickle.HIGHEST_PROTOCOL)
            idval = checkpoint_recent_fs.put(blob, **val_dict)
            msg = 'Saved recent (with filters) to id %s'
            print(msg % str(idval))
    

# GPU Model interface
class IGPUModel:
    def __init__(self, model_name, op, load_dic, filename_options=[], dp_params={}):
        # these are input parameters
        self.model_name = model_name
        self.op = op
        self.options = op.options
        self.load_dic = load_dic
        self.filename_options = filename_options
        self.device_ids = self.op.get_value('gpu')
        self.fill_excused_options()
        self.checkpoint_writer = None
        #assert self.op.all_values_given()
        
        for o in op.get_options_list():
            setattr(self, o.name, o.value)

        self.dp_params = dp_params
        self.loaded_from_checkpoint = load_dic is not None
        # these are things that the model must remember but they're not input parameters
        if self.loaded_from_checkpoint:
            self.model_state = load_dic["model_state"]
            if not "experiment_data" in self.options or not self.options["experiment_data"].value_given:
                self.experiment_data = load_dic["rec"]["experiment_data"]
        else:
            self.model_state = collections.OrderedDict()
            self.model_state["train_outputs"] = []
            self.model_state["test_outputs"] = []
            self.model_state["epoch"] = 1
            self.model_state["batchnum"] = self.train_batch_range[0]
            if not self.options["experiment_data"].value_given:
                idval = model_name + "_" + '_'.join(['%s_%s' % (char, self.options[opt].get_str_value()) for opt, char in filename_options]) + '_' + strftime('%Y-%m-%d_%H.%M.%S')
                self.experiment_data = collections.OrderedDict([('experiment_id', idval)])

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
        try:
            self.test_data_provider = DataProvider.get_instance(self.data_path, self.test_batch_range,
                                                                type=self.dp_type, dp_params=self.dp_params, test=True)
            self.train_data_provider = DataProvider.get_instance(self.data_path, self.train_batch_range,
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
        else:
            self.train()
        self.cleanup()
        if self.force_save:
            self.save_state().join()
        sys.exit(0)
    
    def train(self):
        print "============================================"
        print "Training %s" % self.model_name
        self.op.print_values()
        print "========================="
        self.print_model_state()
        print "Running on CUDA device(s) %s" % ", ".join("%d" % d for d in self.device_ids)
        print "Current time: %s" % asctime(localtime())
        print "========================="

        if self.save_initial and self.get_num_batches_done() == 1:
            self.test_outputs += [self.get_test_error()]
            self.conditional_save()

        next_data = self.get_next_batch()
        while self.epoch <= self.num_epochs:
            data = next_data
            self.epoch, self.batchnum = data[0], data[1]
            self.print_iteration()
            sys.stdout.flush()
            
            compute_time_py = time()
            self.start_batch(data)
            
            # load the next batch while the current one is computing
            t0 = time()
            next_data = self.get_next_batch()
            t1 = time()
            print('T1-T0', t1 - t0)
            batch_output = self.finish_batch()
            t2 = time()
            print('T2-T1', t2 - t1)
            self.train_outputs += [batch_output]
            self.print_train_results()

            if self.get_num_batches_done() % self.testing_freq == 0:
                self.sync_with_host()
                self.test_outputs += [self.get_test_error()]
                self.print_test_results()
                self.print_test_status()
                self.conditional_save()
            
            self.print_elapsed_time(time() - compute_time_py)
    
    def cleanup(self):
        if self.checkpoint_writer is not None:
            self.checkpoint_writer.join()
            self.checkpoint_writer = None
        
    def print_model_state(self):
        pass
    
    def get_num_batches_done(self):
        return len(self.train_batch_range) * (self.epoch - 1) + self.batchnum - self.train_batch_range[0] + 1
    
    def get_next_batch(self, train=True):
        dp = self.train_data_provider
        if not train:
            dp = self.test_data_provider
        return self.parse_batch_data(dp.get_next_batch(), train=train)
    
    def parse_batch_data(self, batch_data, train=True):
        return batch_data[0], batch_data[1], batch_data[2]['data']
    
    def start_batch(self, batch_data, train=True):
        self.libmodel.startBatch(batch_data[2], not train)
    
    def finish_batch(self):
        return self.libmodel.finishBatch()
    
    def print_iteration(self):
        print "\t%d.%d..." % (self.epoch, self.batchnum),
    
    def print_elapsed_time(self, compute_time_py):
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
        
    def sync_with_host(self):
        if self.checkpoint_writer is not None:
            self.checkpoint_writer.join()
            self.checkpoint_writer = None
        self.libmodel.syncWithHost()
        
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
            start_time_test = time()
            self.start_batch(data, train=False)
            load_next = (not self.test_one or self.test_only) and data[1] < self.test_batch_range[-1]
            if load_next: # load next batch
                next_data = self.get_next_batch(train=False)
            test_outputs += [self.finish_batch()]
            if self.test_only: # Print the individual batch results for safety
                print "batch %d: %s" % (data[1], str(test_outputs[-1])),
                self.print_elapsed_time(time() - start_time_test)
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
        
        dic = collections.OrderedDict([("model_state", self.model_state),
                           ("op", self.op)])
            
        assert self.checkpoint_writer is None
        self.checkpoint_writer = CheckpointWriter(dic,
                                                  self.checkpoint_fs_host,
                                                  self.checkpoint_fs_port,
                                                  self.checkpoint_db_name,
                                                  self.checkpoint_fs_name,
                                                  self.save_filters,
                                                  self.save_recent_filters,
                                                  self.saving_freq,
                                                  self.testing_freq,
                                                  self.epoch,
                                                  self.batchnum,
                                                  self.get_num_batches_done(),
                                                  self.experiment_data
                                                  )
        self.checkpoint_writer.start()
        return self.checkpoint_writer
        
    def get_progress(self):
        num_batches_total = self.num_epochs * len(self.train_batch_range)
        return min(1.0, max(0.0, float(self.get_num_batches_done()-1) / num_batches_total))
    
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
        count = checkpoint_fs._GridFS__files.find(query).count()
        fs_to_use = checkpoint_fs
        rec = None
        loading_from = 0
        if count > 0:
            rec = checkpoint_fs._GridFS__files.find(query, sort=[('timestamp', -1)])[0]

        checkpoint_recent_fs = get_recent_checkpoint_fs(checkpoint_fs_host,
                                          checkpoint_fs_port,
                                          checkpoint_db_name,
                                          checkpoint_fs_name)
        count_recent = checkpoint_recent_fs._GridFS__files.find(query).count()
        if count_recent > 0:
            rec_r = checkpoint_recent_fs._GridFS__files.find(query, sort=[('timestamp', -1)])[0]
            if rec is None or rec_r['timestamp'] > rec['timestamp']:
                loading_from = 1
                rec = rec_r
                fs_to_use = checkpoint_recent_fs
                    
        if (count + count_recent) == 0:
            raise NoMatchingCheckpointError('No Matching Checkpoint for query %s in db %s, %s, %s, %s' % (repr(query), checkpoint_fs_host, checkpoint_fs_port, checkpoint_db_name, checkpoint_fs_name))
       
        if loading_from == 0:
            print('Loading checkpoint from regular storage.')
        else:
            print('Loading checkpoint from "recent" storage.')
 
        if not only_rec:
            load_dic = cPickle.loads(fs_to_use.get_last_version(_id=rec['_id']).read())
        else:
            load_dic = {}
        load_dic['rec'] = rec
        return load_dic

    @staticmethod
    def get_options_parser():
        op = OptionsParser()
        op.add_option("load-file", "load_file", StringOptionParser, "Load file", default="", excuses=OptionsParser.EXCUSE_ALL)
        op.add_option("train-range", "train_batch_range", RangeOptionParser, "Data batch range: training")
        op.add_option("test-range", "test_batch_range", RangeOptionParser, "Data batch range: testing")
        op.add_option("data-provider", "dp_type", StringOptionParser, "Data provider", default="default")
        op.add_option("test-freq", "testing_freq", IntegerOptionParser, "Testing frequency", default=25)
        op.add_option("epochs", "num_epochs", IntegerOptionParser, "Number of epochs", default=500)
        op.add_option("data-path", "data_path", StringOptionParser, "Data path")
        
        op.add_option("max-test-err", "max_test_err", FloatOptionParser, "Maximum test error for saving")
        op.add_option("test-only", "test_only", BooleanOptionParser, "Test and quit?", default=0)
        op.add_option("test-one", "test_one", BooleanOptionParser, "Test on one batch at a time?", default=1)
        op.add_option("save-initial", "save_initial", BooleanOptionParser, "Save initial state as checkpoint before training?", default=0)
        op.add_option("force-save", "force_save", BooleanOptionParser, "Force save before quitting", default=0)
        op.add_option("gpu", "gpu", ListOptionParser(IntegerOptionParser), "GPU override")
        ####### db configs #######
        op.add_option("save-db", "save_db", BooleanOptionParser, "Save checkpoints to mongo database?", default=0)
        op.add_option("save-filters", "save_filters", BooleanOptionParser, "Save filters to database?", default=1)
        op.add_option("save-recent-filters", 'save_recent_filters', BooleanOptionParser, "Save recent filters to database?", default=1)
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
        op.add_option("load-query", "load_query", JSONOptionParser, "Query for loading checkpoint from database", default="", excuses=OptionsParser.EXCUSE_ALL)        
        
        return op

    @staticmethod
    def print_data_providers():
        print "Available data providers:"
        for dp, desc in dp_types.iteritems():
            print "    %s: %s" % (dp, desc)
            

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
                print "Loading checkpoint from database."
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


def get_convenient_mongodb_representation_base(op, model_state):
    val_dict = collections.OrderedDict([(_o.name, _o.value) for _o in op.get_options_list()])
    def make_mongo_safe(_d):
        for _k in _d:
            if '.' in _k:
                _d[_k.replace('.', '___')] = _d.pop(_k)
    if 'load_query' in val_dict:
        val_dict['load_query'] = make_mongo_safe(val_dict['load_query'])

    #import cPickle
    #with open('testdump.pkl', 'wb') as _f:
    #    cPickle.dump(model_state, _f)

    for k in model_state:
        if k == 'layers':
            bad_keys = ['weights', 'inputLayers', 'biasesInc',
                        'biases', 'weightsInc']
            layers = copy.deepcopy(model_state[k])
            for _l in layers:
                for _bk in bad_keys:
                    if _bk in layers[_l]:
                        layers[_l].pop(_bk)
            node_order = get_node_order(layers)
            layers = collections.OrderedDict([(_l, layers[_l]) for _l in node_order])
            val_dict[k] = layers
        elif k == 'train_outputs':
            tfreq = val_dict['testing_freq']
            val_dict[k] = model_state[k][-tfreq:]
        elif k == 'test_outputs':
            val_dict[k] = model_state[k][-1]
        else:
            val_dict[k] = model_state[k]

    return SONify(val_dict)


import networkx as nx
import itertools
def get_node_order(layers):
    G = nx.DiGraph()
    nodes = [x['name'] for x in layers.values()]
    edges = list(itertools.chain(*[[(y['name'], x['name']) for y in x.get('inputLayers', [])] for x in layers.values()]))
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    return nx.topological_sort(G)
