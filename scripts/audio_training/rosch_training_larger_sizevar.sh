#!/bin/bash

source ~/.bash_profile 
python /home/yamins/make_tunnel.py
python /home/yamins/make_tunnel_27017.py

cd /om/user/yamins/src/archconvnets/archconvnets/convnet2/

python convnet.py --save-initial=0 --data-path=/om/user/yamins/.skdata/rosch_batches256 --crop=9 --test-range=4401-4675 --train-range=0-4400 --epochs=40 --layer-def=/om/user/yamins/src/archconvnets/archconvnets/convnet2/layers/audio_models/layers-larger-rosch-singlelv.cfg --layer-params=/om/user/yamins/src/archconvnets/archconvnets/convnet2/layers/audio_models/layer-params-larger-singlelv.cfg --data-provider=general-cropped --test-freq=500 --saving-freq=8 --conserve-mem=1 --gpu=1 --checkpoint-fs-port=29101 --checkpoint-db-name=roschinet_training_0 --checkpoint-fs-name='models'  --experiment-data='{"experiment_id": "rosch_training_larger_sizevar"}' --dp-params='{"num_batches_for_mean": 50, "perm_type": "random", "perm_seed": 0, "preproc": {"normalize": false, "dtype": "float32", "resize_to": [256, 256, 3], "mode": "RGB", "crop": null, "mask": null}, "batch_size": 256, "meta_attribute": "s", "dataset_name": ["dldata.stimulus_sets.synthetic.synthetic_datasets", "RoschDataset"]}'