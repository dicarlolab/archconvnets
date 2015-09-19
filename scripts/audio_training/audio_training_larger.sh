#!/bin/bash

source ~/.bash_profile
python /home/yamins/make_tunnel.py
cd /om/user/yamins/src/archconvnets/archconvnets/convnet2/

python convnet.py --img-flip=0 --save-initial=1 --data-path=/om/user/yamins/.skdata/audio_batches_newSnrs --crop=0 --test-range=4201-4520 --train-range=0-4200 --layer-def=/om/user/yamins/src/archconvnets/archconvnets/convnet2/layers/audio_models/layers-larger.cfg --layer-params=/om/user/yamins/src/archconvnets/archconvnets/convnet2/layers/audio_models/layer-params-larger.cfg --data-provider=general-cropped --test-freq=500 --saving-freq=8 --conserve-mem=1 --gpu=0 --checkpoint-fs-port=29101 --checkpoint-db-name=audio_training_1audio_training_large --checkpoint-fs-name='models'  --experiment-data='{"experiment_id": "audio_training_newSnrs_larger_fix"}' --dp-params='{"num_batches_for_mean": 50, "perm_type": "random", "perm_seed": 0, "preproc": {"normalize": false, "dtype": "float32", "resize_to": [225, 225], "mode": "RGB", "crop": null, "mask": null}, "batch_size": 256, "meta_attribute": "word_formatted", "dataset_name": ["dldataAudio.stimulus_sets.combined_datasets", "CombinedTrain0Balanced_newSnrs"]}'
