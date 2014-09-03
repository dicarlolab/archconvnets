#!/bin/bash

source ~/.bash_profile

cd /om/user/yamins/src/archconvnets/archconvnets/convnet2/

python convnet.py --data-path=/om/user/yamins/.skdata/audio_batches_balanced_1 --crop=0 --test-range=4801-5110 --train-range=0-4800 --layer-def=/om/user/yamins/src/archconvnets/archconvnets/convnet2/layers/audio_models/layers-imagenet-1gpu.cfg --layer-params=/om/user/yamins/src/archconvnets/archconvnets/convnet2/layers/layer-params-imagenet-1gpu.cfg --data-provider=general-cropped --test-freq=250 --saving-freq=2 --conserve-mem=1 --gpu=0 --checkpoint-fs-port=29101 --checkpoint-db-name=audio_test_2 --checkpoint-fs-name='models'  --experiment-data='{"experiment_id": "audio_training"}' --dp-params='{"num_batches_for_mean": 50, "perm_type": "random", "perm_seed": 0, "preproc": {"normalize": false, "dtype": "float32", "resize_to": [225, 225], "mode": "RGB", "crop": null, "mask": null}, "batch_size": 256, "meta_attribute": "word_formatted", "dataset_name": ["dldataAudio.stimulus_sets.combined_datasets", "CombinedTrain0Balanced"]}'
