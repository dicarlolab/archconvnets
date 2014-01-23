from archconvnets.convnet.convnet import ConvNet, IGPUModel
import numpy as np
dp_params = {'dataset_name': ('dldata.stimulus_sets.synthetic.synthetic_datasets', 'TrainingDataset'),
'batch_size': 128,
'meta_attribute': 'obj',
'perm_type': 'random',
'perm_seed': 0,
'preproc': {'resize_to': (128, 128, 3),
'mode': 'RGB',
'dtype': 'float32',
'normalize': False}
}

nr = np.RandomState()

if __name__ == "main":
    op = ConvNet.get_options_parser()
    op, load_dic = IGPUModel.parse_options(op)
    nr.seed(op.options['random_seed'].value)
    model = ConvNet(op, load_dic, dp_params=dp_params)
    model.start()

"""
Use this command to run (see Alex's documenatation for explanation of these parameters)
python run_synthetic_training.py --data-path=/export/imgnet_storage_full/yamins_skdata/sythetic_batches_0 --crop=7 --save-path=/export/imgnet_storage_full/ --test-range=950-999 --train-range=0-949 --layer-def=/home/yamins/archconvnets/archconvnets/convnet/ut_model_full/layer_nofc_0.cfg --layer-params=/home/darren/archconvnets/archconvnets/convnet/ut_model_full/layer-params.cfg --data-provider=general-cropped --test-freq=50 --conserve-mem=1 --max-filesize=99999999 --img-size=128
"""