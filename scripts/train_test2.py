
import os

def do_extract():
    cmd_tmpl  = """python extractnet.py --gpu=2 --test-range=%s --train-range=0 --data-provider=general-cropped --feature-layer=%s --write-disk=1 --feature-path=/export/storage/yamins_skdata/features/%s_%s --data-path=/export/storage/yamins_skdata/%s --load-query='%s' --checkpoint-fs-port=%d --checkpoint-db-name=%s --checkpoint-fs-name=%s --dp-params='{"perm_type": "random", "perm_seed": 0, "preproc": {"normalize": false, "dtype": "float32", "resize_to": %s, "mode": "RGB", "crop": null, "mask": null}, "batch_size": 256, "meta_attribute": "%s", "dataset_name": ["%s", "%s"]}'"""

    layer_names = [#'data',
                   #'conv1_1a', 'conv1_1b', 
                   #'conv2_4a', 'conv2_4b', 
                   #'conv3_7a', 'conv3_7b', 
                   #'conv4_8a', 'conv4_8b', 
                   #'conv5_9a', 'conv5_9b',
                   #'pool1_3a', 'pool1_3b', 
                   #'pool2_6a', 'pool2_6b', 
                   #'pool3_11a', 'pool3_11b',
                   #'fc1_12a', 'fc1_12b', 
                   #'rnorm4_13a', 'rnorm4_13b',
                   #'fc2_14a', 'fc2_14b',  
                   'rnorm5_15a', 'rnorm5_15b'
        ]
    layer_names = [','.join(layer_names)]


    #dsetmods = ['dldata.stimulus_sets.hvm', 'dldata.stimulus_sets.synthetic.synthetic_datasets']
    #dsetobjs = ['HvMWithDiscfade', 'TrainingDatasetLarge']
    data_paths = [
                   #"hvm_batches_138_permuted", 
                   #"synthetic_batches_large",
                  #"imagenet_challenge_138"
                  #"sketch_batches",
                  #"sketch3d_batches",
                  #"sketch3d_2_bacthes", 
                  #"sketch3d_3_batches",
                  "sketch3_inetequiv_batches",
                  #"cvcl_mm_PNAS2008_UniqueObjects_batches",
                  #"cvcl_mm_PNAS2008_StatePairs_batches",
                  #"cvcl_mm_PNAS2008_ExemplarPairs_batches"     
                 ]
    resize_tos = [#[138, 138]',
                  #'[138, 138, 3]',
                  '[138, 138]'
                  ] 
    batch_limits = ['0-796',
                    #'0-something'
                    ]
    dsetmods = [#'sketchloop.python.datasets',
                'sketchloop.python.datasets'
                #'dldata.stimulus_sets.cvcl_mm',
               ]
    dsetobjs = [#'Siggraph2012Sketches', 
                #'ThreeDModelEquivalentSiggraph2012Sketches2',
                #'ThreeDModelEquivalentSiggraph2012Sketches3',
                'ImagenetEquivalentSiggraph2012Sketches3',
               #'PNAS2008_UniqueObjects', 
               #'PNAS2008_StatePairs', 
               #'PNAS2008_ExemplarPairs'
                ]
    mattrs = [#'category', 
              'synset']

    model_names = ["imagenet_trained",  
                   #"rosch_trained",
                   #'imagenet_trained_nofc'
                   ]
    queries = ['{"experiment_data.experiment_id":"imagenet_training_reference_0"}', 
               #'{"experiment_data.experiment_id": "synthetic_training_rosch_category"}',
              ]
    fs_ports = [29101]
    db_names = ['reference_models']
    fs_names = ['models', 
                ]

    vals = [(batch_limit, layer_name, model_name, dsetobj, data_path, query, fs_port, db_name, fs_name, rst, mattr, dsetmod, dsetobj) for batch_limit, data_path, rst, mattr, dsetmod, dsetobj in zip(batch_limits, data_paths, resize_tos, mattrs, dsetmods, dsetobjs) for model_name, query, fs_port, db_name, fs_name in zip(model_names, queries, fs_ports, db_names, fs_names) for layer_name in layer_names]

    for val in vals[:1]:
        print('VAL', val)
        cmd = cmd_tmpl % val
        os.system(cmd)
        print(cmd)


import filter_analysis
import cPickle
def do_train(outdir):
    model_names = ["imagenet_trained", "synthetic_category_trained", "rosch_trained"]
    layer_names = ['conv1_1a', 'conv1_1b', 'conv2_4a', 'conv2_4b', 'conv3_7a', 'conv3_7b', 'conv4_8a', 'conv4_8b', 'conv5_9a', 'conv5_9b', 'pool3_11a', 'pool3_11b', 'fc1_12a', 'fc1_12b', 'rnorm4_13a', 'rnorm4_13b','fc2_14a', 'fc2_14b',  'rnorm5_15a', 'rnorm5_15b']
    for mname in model_names[2:]:
        for lname in layer_names[:]:
            res = filter_analysis.compute_performance(mname, lname)
            fname = os.path.join(outdir, '%s_%s.pkl' %(mname, lname))
            with open(fname, 'w') as _f:
                cPickle.dump(res, _f)

    model_names = ["imagenet_trained_nofc"]
    layer_names = ['conv1_1a', 'conv1_1b', 'conv2_4a', 'conv2_4b', 'conv3_7a', 'conv3_7b', 'conv4_8a', 'conv4_8b', 'conv5_9a', 'conv5_9b', 'pool3_11a', 'pool3_11b']
    for mname in model_names[:1]:
        for lname in layer_names[:1]:
            res = filter_analysis.compute_performance(mname, lname)
            fname = os.path.join(outdir, '%s_%s.pkl' %(mname, lname))
            with open(fname, 'w') as _f:
                cPickle.dump(res, _f)

if __name__ == '__main__':
    do_extract()
