import os
cmd_tmpl  = """python extractnet.py --test-range=0-%d --train-range=0 --data-provider=general-cropped --write-features=%s --feature-path=/export/storage/yamins_skdata/features/%s_%s_%s --data-path=/export/storage/yamins_skdata/%s --load-query='%s' --checkpoint-fs-name=%s --dp-params='{"perm_type": "random", "perm_seed": 0, "preproc": {"normalize": false, "dtype": "float32", "resize_to": %s, "mode": "RGB"}, "batch_size": 256, "meta_attribute": "category", "dataset_name": ["%s", "%s"]}'"""

layer_names = ['conv1_1a', 'conv1_1b', 'conv2_4a', 'conv2_4b', 'conv3_7a', 'conv3_7b', 'conv4_8a', 'conv4_8b', 'conv5_9a', 'conv5_9b']
layer_names += ['pool3_11a', 'pool3_11b']
#layer_names += ['fc1_12a', 'fc1_12b', 'rnorm4_13a', 'rnorm4_13b','fc2_14a', 'fc2_14b',  'rnorm5_15a', 'rnorm5_15b']
#model_names = ["imagenet_trained", "synthetic_category_trained"]
model_names = ['imagenet_trained_nofc']
data_paths = ["hvm_batches_138_permuted", "synthetic_batches_large"]
resize_tos = ['[138, 138]', '[138, 138, 3]']
batch_limits = [22, 50]
#queries = ['{"experiment_data.experiment_id":"imagenet_training_reference_0"}', '{"experiment_data.experiment_id": "synthetic_training_bsize256_large_category"}']
queries = ['{"experiment_data.experiment_id":"imagenet_training_reference_0_nofc"}']
#fs_names = ['reference_models', 'convnet_checkpoint_fs']
fs_names = ['reference_models']
dsetmods = ['dldata.stimulus_sets.hvm', 'dldata.stimulus_sets.synthetic.synthetic_datasets']
dsetobjs = ['HvMWithDiscfade', 'TrainingDatasetLarge']

vals = [(batch_limit, layer_name, model_name, dsetobj, layer_name, data_path, query, fs_name, rst, dsetmod, dsetobj) for batch_limit, data_path, rst, dsetmod, dsetobj in zip(batch_limits, data_paths, resize_tos, dsetmods, dsetobjs) for model_name, query, fs_name in zip(model_names, queries, fs_names) for layer_name in layer_names]

for val in vals[1:]:
    print('VAL', val)
    cmd = cmd_tmpl % val
    os.system(cmd)
    print(cmd)
