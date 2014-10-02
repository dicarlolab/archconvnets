#!/bin/bash

source ~/.bash_profile
python /home/yamins/make_tunnel.py

cd /om/user/yamins/src/archconvnets/archconvnets/convnet2/

python convnet.py --save-initial=1 --data-path=/om/user/yamins/.skdata/audio_batches_balanced_1 --crop=0 --test-range=4401-5110 --train-range=0-4400 --layer-def=/om/user/yamins/src/archconvnets/archconvnets/convnet2/layers/audio_models/layers-standard-l3.cfg --layer-params=/om/user/yamins/src/archconvnets/archconvnets/convnet2/layers/audio_models/layer-params-standard.cfg --data-provider=general-cropped --test-freq=250 --saving-freq=4 --conserve-mem=1 --gpu=0 --checkpoint-fs-port=29101 --checkpoint-db-name=audio_training_0 --checkpoint-fs-name='models'  --experiment-data='{"experiment_id": "audio_training_l3"}' --dp-params='{"num_batches_for_mean": 50, "perm_type": "random", "perm_seed": 0, "preproc": {"normalize": false, "dtype": "float32", "resize_to": [225, 225], "mode": "RGB", "crop": null, "mask": null}, "batch_size": 256, "meta_attribute": "word_formatted", "dataset_name": ["dldataAudio.stimulus_sets.combined_datasets", "CombinedTrain0Balanced"]}'
