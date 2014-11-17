#!/bin/bash

source ~/.bash_profile
python /home/yamins/make_tunnel.py
cd /om/user/yamins/src/archconvnets/archconvnets/convnet2/

python convnet.py --checkpoint-fs-port=29101 --checkpoint-db-name=audio_training_0 --checkpoint-fs-name='models' --load-query='{"experiment_data.experiment_id": "audio_training_newSnrs_larger_fix"}' --epochs=40 --layer-params=/om/user/yamins/src/archconvnets/archconvnets/convnet2/layers/audio_models/layer-params-larger-fc6-speaker-eval1.cfg --layer-def=/om/user/yamins/src/archconvnets/archconvnets/convnet2/layers/audio_models/layers-larger-fc6-speaker-eval.cfg --experiment-data='{"experiment_id": "audio_training_newSnrs_larger_fix_fc6_speaker_eval"}' --dp-params='{"num_batches_for_mean": 25, "perm_type": "random", "perm_seed": 0, "preproc": {"normalize": false, "dtype": "float32", "resize_to": [225, 225], "mode": "RGB", "crop": null, "mask": null}, "batch_size": 256, "meta_attribute": "speaker", "dataset_name": ["dldataAudio.stimulus_sets.combined_datasets", "CombinedTrain0Balanced_newSnrs4"]}' --gpu=2 --save-initial=0 --data-path=/om/user/yamins/.skdata/audio_batches_newSnrs4 --start-epoch=1 --start-batch=0
