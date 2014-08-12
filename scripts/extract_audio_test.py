
import os

def do_extract():
    n_gpu = 1 #os.environ.get('EXTRACTION_NGPU', 0)
    cmd_tmpl  = """python extractnet.py --gpu=%d --test-range=%s --train-range=0 --data-provider=general-cropped --feature-layer=%s --write-disk=1 --feature-path=/export/storage/yamins_skdata/features/%s_%s --data-path=/export/storage/yamins_skdata/%s --load-query='%s' --checkpoint-fs-port=%d --checkpoint-db-name=%s --checkpoint-fs-name=%s --dp-params='{"perm_type": "random", "perm_seed": 0, "preproc": {"normalize": false, "dtype": "float32", "resize_to": %s, "mode": "RGB", "crop": null, "mask": null}, "batch_size": 256, "meta_attribute": "%s", "dataset_name": ["%s", "%s"]}'"""

    layer_names = [
                   #'data',
                   #'pool1', 
                   #'pool2', 
                   #'conv3_neuron',
                   #'conv4_neuron',
                   'fc6'
                   ]
    layer_names = [','.join(layer_names)]


    data_paths = [
                   #"imagenet_challenge_256",
                   'audio_batches_1'
                 ]
    resize_tos = [
                  '[225, 225]',
                  ]

    batch_limits = [
                    '0-1'
                    ]
    dsetmods = [
                'dldataAudio.stimulus_sets.testset'
                ]
    dsetobjs = [
                'TimitTesting0'
                ]
    mattrs = [ 'word_formatted'
                ]

    model_names = ["audio_trained_0", 
                   
                   ]
    queries = ['{"experiment_data.experiment_id":"audio_training"}', 
              ]
    fs_ports = [29101]
    db_names = ['audio_test_0']
    fs_names = [
                'models', 
                ]

    vals = [(batch_limit, layer_name, model_name, dsetobj, data_path, query, fs_port, db_name, fs_name, rst, mattr, dsetmod, dsetobj) for batch_limit, data_path, rst, mattr, dsetmod, dsetobj in zip(batch_limits, data_paths, resize_tos, mattrs, dsetmods, dsetobjs) for model_name, query, fs_port, db_name, fs_name in zip(model_names, queries, fs_ports, db_names, fs_names) for layer_name in layer_names]

    for val in vals[:]:
        print('VAL', val)
        cmd = cmd_tmpl % ((int(n_gpu), ) + val)
        os.system(cmd)
        print(cmd)


if __name__ == '__main__':
    do_extract()
