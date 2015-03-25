#!/bin/bash

source ~/.bash_profile 
python /home/yamins/make_tunnel.py
python /home/yamins/make_tunnel_27017.py

cd /om/user/yamins/src/archconvnets/archconvnets/convnet2/

python convnet.py --checkpoint-fs-port=29101 --checkpoint-db-name=roschinet_training_0 --checkpoint-fs-name='models' --load-query='{"experiment_data.experiment_id": "rosch_training_larger_sizevar"}' --gpu=2
