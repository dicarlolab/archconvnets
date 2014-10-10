#!/bin/bash

source ~/.bash_profile 
python /home/yamins/make_tunnel.py
python /home/yamins/make_tunnel_27017.py

cd /om/user/yamins/src/archconvnets/archconvnets/convnet2/

python convnet.py --load-query='{"experiment_data.experiment_id": "imagenet_training_larger_nomean"}' --checkpoint-fs-port=29101 --checkpoint-db-name=inet_training_0 --checkpoint-fs-name='models' --epochs=48
