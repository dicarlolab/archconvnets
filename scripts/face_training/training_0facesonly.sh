#!/bin/bash

source ~/.bash_profile
python /home/yamins/make_tunnel.py
python /home/yamins/make_tunnel_27017.py

cd /om/user/yamins/src/archconvnets/archconvnets/convnet2/

python convnet.py --save-initial=0 --data-path=/om/user/yamins/.skdata/combined_random_256 --crop=4 --train-range=0-5500 --test-range=5501-5928 --epochs=100 --layer-def=/om/user/yamins/src/archconvnets/archconvnets/convnet2/layers/face_models/layers-face-model-bu4dfe-2gpu.cfg --layer-params=/om/user/yamins/src/archconvnets/archconvnets/convnet2/layers/face_models/layer-params-imagenet-bu4dfe-2gpu-model.cfg --data-provider=general-cropped --test-freq=1000 --saving-freq=8 --conserve-mem=1 --gpu=2,3 --checkpoint-fs-port=29101 --checkpoint-db-name=face_training --checkpoint-fs-name='models'  --experiment-data='{"experiment_id": "combined_bu4dfe_only_faceid_emotion"}' --dp-params='{"num_batches_for_mean": 100, "perm_type": "random", "perm_seed": 0, "batch_size": 256, "meta_attribute": ["face_id", "emotion"], "dataset_name": ["dldata.stimulus_sets.face_project.datasets", "CombinedLargeDataset"], "preproc": [{"normalize": false, "dtype": "float32", "resize_to": [256, 256, 3], "mode": "RGB", "crop": null, "mask": null, "noise": {"magnitude": 0.4, "smoothing": 0.75}}, {"normalize": false, "dtype": "float32", "resize_to": [256, 256, 3], "mode": "RGB", "crop": null, "mask": null}], "subslice": ["BU4DFE", {}]}'