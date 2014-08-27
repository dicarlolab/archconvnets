#!/bin/bash

source ~/.bash_profile 

cd /om/user/yamins/src/archconvnets/archconvnets/convnet2/

python convnet.py --load-query='{"experiment_data.experiment_id":"audio_training"}' --checkpoint-fs-port=29101 --checkpoint-db-name=audio_test_2 --checkpoint-fs-name='models'
