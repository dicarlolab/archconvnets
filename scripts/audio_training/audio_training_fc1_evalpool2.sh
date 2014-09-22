#!/bin/bash

source ~/.bash_profile 
python /home/yamins/make_tunnel.py
cd /om/user/yamins/src/archconvnets/archconvnets/convnet2/

python convnet.py --load-query='{"experiment_data.experiment_id":"audio_training_fc1"}' --checkpoint-fs-port=29101 --checkpoint-db-name=audio_training_0 --checkpoint-fs-name='models' --layer-def='/om/user/yamins/src/archconvnets/archconvnets/convnet2/layers/audio_models/layers-standard-fc1-pool2eval.cfg' --layer-params='/om/user/yamins/src/archconvnets/archconvnets/convnet2/layers/audio_models/layer-params-standard-pool2eval.cfg' --experiment-data='{"experiment_id": "audio_training_fc1_evalpool2"}'
