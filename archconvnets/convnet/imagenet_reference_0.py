import pymongo as pm
import gpumodel


def attach_model(model):
    if not hasattr(model, 'cached_reference_0'):
        print 'Caching trained model ...'
        query = {'experiment_data.experiment_id': 'imagenet_training_reference_0'}
        checkpoint_fs_host = 'localhost'
        checkpoint_fs_port = 27017
        checkpoint_db_name = 'convnet_checkpoint_db'
        checkpoint_fs_name = 'reference_models'
        model.cached_reference_0 = gpumodel.IGPUModel.load_checkpoint_from_db(query,
                                                                              checkpoint_fs_host,
                                                                              checkpoint_fs_port,
                                                                              checkpoint_db_name,
                                                                              checkpoint_fs_name, only_rec=False)


def layerWeightsAndInc(name, idx, shape, params, model):
    layer_name = params[0]
    attach_model(model)
    layer_list = model.cached_reference_0['model_state']['layers']
    layer_names = [l['name'] for l in layer_list]
    layer_dic = layer_list[layer_names.index(layer_name)]
    return layer_dic['weights'][idx]#, layer_dic['weightsInc'][idx]


def layerBiasesAndInc(name, shape, params, model):
    layer_name = params[0]
    attach_model(model)
    layer_list = model.cached_reference_0['model_state']['layers']
    layer_names = [l['name'] for l in layer_list]
    layer_dic = layer_list[layer_names.index(layer_name)]
    return layer_dic['biases']#, layer_dic['biasesInc']
