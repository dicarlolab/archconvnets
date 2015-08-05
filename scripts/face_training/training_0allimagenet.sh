#!/bin/bash

source ~/.bash_profile
python /home/yamins/make_tunnel.py
python /home/yamins/make_tunnel_27017.py

cd /om/user/yamins/src/archconvnets/archconvnets/convnet2/

python convnet.py --save-initial=0 --data-path=/om/user/yamins/.skdata/combined_random_256 --crop=4 --train-range=0-10962 --test-range=10963-11962 --epochs=100 --layer-def=/om/user/yamins/src/archconvnets/archconvnets/convnet2/layers/face_models/layers-face-model-allimagenet-2gpu.cfg --layer-params=/om/user/yamins/src/archconvnets/archconvnets/convnet2/layers/face_models/layer-params-imagenet-2gpu-model.cfg --data-provider=general-cropped --test-freq=1000 --saving-freq=8 --conserve-mem=1 --gpu=0,1 --checkpoint-fs-port=29101 --checkpoint-db-name=face_training --checkpoint-fs-name='models'  --experiment-data='{"experiment_id": "combined_allimagenet"}' --dp-params='{"num_batches_for_mean": 100, "perm_type": "random", "perm_seed": 0, "batch_size": 256, "meta_attribute": "category", "dataset_name": ["dldata.stimulus_sets.face_project.datasets", "CombinedLargeDataset"], "preproc": [{"normalize": false, "dtype": "float32", "resize_to": [256, 256, 3], "mode": "RGB", "crop": null, "mask": null, "noise": {"magnitude": 0.4, "smoothing": 0.75}}, {"normalize": false, "dtype": "float32", "resize_to": [256, 256, 3], "mode": "RGB", "crop": null, "mask": null}], "subslice": ["all_imagenet", {}]}'