#!/bin/bash

source ~/.bash_profile
python /home/yamins/make_tunnel.py
python /home/yamins/make_tunnel_27017.py

cd /om/user/yamins/src/archconvnets/archconvnets/convnet2/

python convnet.py --gpu=2,3 --checkpoint-fs-port=29101 --checkpoint-db-name=face_training --checkpoint-fs-name='models'  --load-query='{"experiment_data.experiment_id": "combined_bu4dfe_only_faceid_emotion"}' 
