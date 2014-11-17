#!/bin/bash

source ~/.bash_profile
python /home/yamins/make_tunnel.py
cd /om/user/yamins/src/archconvnets/archconvnets/convnet2/

python convnet.py --checkpoint-fs-port=29101 --checkpoint-db-name=audio_training_0 --checkpoint-fs-name='models' --load-query='{"experiment_data.experiment_id": "audio_training_newSnrs4_larger"}' --epochs=40
