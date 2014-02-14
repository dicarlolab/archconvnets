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

class ExtractNetError(Exception):
    pass

class ExtractConvNet(ConvNet):
    def __init__(self, op, load_dic, dp_params=None):
        ConvNet.__init__(self, op, load_dic, dp_params=dp_params)

    def get_gpus(self):
        self.need_gpu = True
        if self.need_gpu:
            ConvNet.get_gpus(self)

    def init_data_providers(self):
        class Dummy:
            def advance_batch(self):
                pass
        if self.need_gpu:
            ConvNet.init_data_providers(self)
        else:
            self.train_data_provider = self.test_data_provider = Dummy()

    def import_model(self):
        if self.need_gpu:
            ConvNet.import_model(self)

    def init_model_state(self):
        ConvNet.init_model_state(self)
        self.ftr_layer_idxs = [self.get_layer_idx(_l) for _l in self.op.get_value('write_features')]


    def init_model_lib(self):
        if self.need_gpu:
            ConvNet.init_model_lib(self)

    def do_write_features(self):
        next_data = self.get_next_batch(train=False)
        b1 = next_data[1]
        for lnum, lind in enumerate(self.ftr_layer_idxs):
            lname = self.op.get_value('write_features')[lnum]
            ldir = self.feature_path + '_' + lname
            if not os.path.exists(ldir):
                os.makedirs(ldir)
            num_ftrs = self.layers[lind]['outputs']
            while True:
                batch = next_data[1]
                data = next_data[2]
                ftrs = n.zeros((data[0].shape[1], num_ftrs), dtype=n.single)
                self.libmodel.startFeatureWriter(data + [ftrs], lind, 1)

                # load the next batch while the current one is computing
                next_data = self.get_next_batch(train=False)
                self.finish_batch()
                path_out = os.path.join(ldir, 'data_batch_%d' % batch)
                pickle(path_out, {'data': ftrs, 'labels': data[1]})
                print "Wrote feature file %s" % path_out
                if next_data[1] == b1:
                    break
            pickle(os.path.join(ldir, 'batches.meta'), {'source_model':self.load_file,
                                                                 'source_model_query': self.load_query,
                                                                 'num_vis':num_ftrs})

    def start(self):
        self.op.print_values()
        self.do_write_features()

    @classmethod
    def get_options_parser(cls):
        op = ConvNet.get_options_parser()
        for option in list(op.options):
            if option not in ('gpu', 'load_file', 'train_batch_range', 'test_batch_range', 'load_query', 'checkpoint_fs_host', 'checkpoint_fs_port', 'checkpoint_db_name', 'checkpoint_fs_name', 'data_path', 'dp_type', 'dp_params', 'img_size'):
                op.delete_option(option)
        op.add_option("write-features", "write_features", ListOptionParser(StringOptionParser), "Write test data features from given layer", default="", requires=['feature-path'])
        op.add_option("feature-path", "feature_path", StringOptionParser, "Write test data features to this path (to be used with --write-features)", default="")

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
