__author__ = 'ardila'

import importlib
import numpy as np
import os

modulename = 'acuity_experiments.landolt_cs'
attrname = 'LandoltCsWithNoise'
metafield = 'rotation'

module = importlib.import_module(modulename)
dataset_obj = getattr(module, attrname)
dataset = dataset_obj()
n_batches = np.int(np.ceil(len(dataset.meta)/256.)-1)
base_data_path = '/export/storage2/ardila/'
data_path = base_data_path+str(attrname)+'_batches'

command = """python extractnet.py --test-range=0-%s --train-range=0 --data-provider=general-cropped --checkpoint-fs-port=29101 --checkpoint-fs-name=models --checkpoint-db=reference_models --load-query='{"experiment_data.experiment_id": "nyu_model"}' --feature-layer=pool5,fc6 --data-path=%s --dp-params='{"crop_border": 16, "meta_attribute": "%s", "preproc": {"normalize": false, "dtype": "float32", "resize_to": [256, 256], "mode": "RGB", "mask": null, "crop":null}, "batch_size": 256, "dataset_name": ["%s", "%s"]}' --write-db=1 --write-disk=0"""%(n_batches, data_path, metafield, modulename, attrname)

print command
os.system(command)
