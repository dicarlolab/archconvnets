#!/bin/bash

source ~/.bash_profile 
python /home/yamins/make_tunnel.py
python /home/yamins/make_tunnel_27017.py

cd /om/user/yamins/src/archconvnets/archconvnets/convnet2/

python convnet.py --save-initial=0 --data-path=/om/user/yamins/.skdata/normals_stim1 --crop-border=9,0 --test-range=70-80 --train-range=0-70 --epochs=40 --layer-def=/om/user/yamins/src/archconvnets/archconvnets/convnet2/layers/audio_models/layers-larger-rosch-normals.cfg --layer-params=/om/user/yamins/src/archconvnets/archconvnets/convnet2/layers/audio_models/layer-params-larger-normals.cfg --data-provider=general-map-cropped --test-freq=500 --saving-freq=8 --conserve-mem=1 --gpu=1 --checkpoint-fs-port=29101 --checkpoint-db-name=normals_training --checkpoint-fs-name='models'  --experiment-data='{"experiment_id": "normals_training_test"}' --dp-params='{"num_batches_for_mean": 25, "perm_type": null, "perm_seed": 0, "batch_size": 256, "meta_attribute": "obj", "dataset_name": ["dldata.stimulus_sets.synthetic.reasonable_variation", "NaturalMotionDatasetSingleObjectTest1"], "map_preprocs": [{"normalize": false, "dtype": "float32", "resize_to": [256, 256, 3], "mode": "RGB", "crop": null, "mask": null}, {"normalize": false, "dtype": "float32", "resize_to": [64, 64, 3], "mode": "RGB", "crop": null, "mask": null, "shader": ["/om/user/yamins/.skdata/genthor/resources/shaders/vshader_normals_absolute.glsl", "/om/user/yamins/.skdata/genthor/resources/shaders/fshader_normals.glsl"]}], "map_methods": ["get_images", "get_images"]}'