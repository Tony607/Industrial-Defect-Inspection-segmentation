import os
from tensorflow.keras.models import model_from_json
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.tools import freeze_graph
from tensorflow.keras.models import model_from_json, model_from_yaml


def convert_keras_to_pb(path_to_kmodel, out_node):
    """convert keras model (json with hdf5 format) to tensorflow pb format (protobuf format)
    
    Arguments:
        path_to_kmodel (string): path to keras model files, model structure in json format, model weights in hdf5 format 
    
    Return:
        
    
    Notes:
    
    """
    # get model in json and hdf5 format (from tensorflow.keras export)
    model_json = [file for file in os.listdir(path_to_kmodel) if file.endswith('json')][0]
    # print(model_json)
    model_weights = [file for file in os.listdir(path_to_kmodel) if file.endswith('h5')][0]
    # print(model_weights)
    # load the model structure (json file)
    with open(os.path.join(path_to_kmodel, model_json)) as f:
        model_json_string = f.read()
    model = model_from_json(model_json_string)
    
    # load model weights in hdf5 format (from tensorflow.keras export)
    model.load_weights(os.path.join(path_to_kmodel, model_weights))
        
        
    # All new operations will be in test mode from now on
    K.set_learning_phase(0)
    
    checkpoint_prefix = os.path.join(path_to_kmodel, "saved_checkpoint")
    checkpoint_state_name = "checkpoint_state"
    input_graph_name = 'input_graph.pb'
    output_graph_name = 'output_graph.pb'
    
    # Temporary save graph to disk without weights included.
    saver = tf.train.Saver()
    #saver = saver_lib.Saver(write_version=saver_pb2.SaverDef.V2)
    
    checkpoint_path = saver.save(sess=K.get_session(), 
                                 save_path=checkpoint_prefix, 
                                 global_step=0, 
                                 latest_filename=checkpoint_state_name)
    
    # 
    tf.train.write_graph(graph_or_graph_def=K.get_session().graph,
                         logdir=path_to_kmodel,
                         name=input_graph_name)
    
    
    # Embed weights inside the graph and save to disk
    freeze_graph.freeze_graph(input_graph=os.path.join(path_to_kmodel, input_graph_name),
                              input_saver="",
                              input_binary=False,
                              input_checkpoint=checkpoint_path,
                              output_node_names=out_node,
                              restore_op_name='save/restore_all',
                              filename_tensor_name='save/Const:0',
                              output_graph=os.path.join(path_to_kmodel, output_graph_name),
                              clear_devices=False,
                              initializer_nodes=''
                              )
    
    return model



def print_output_node_names(path_to_kmodel):
    """get last few output node names
    
    
    """
    model_json = [file for file in os.listdir(path_to_kmodel) if file.endswith('json')][0]
    model_weights = [file for file in os.listdir(path_to_kmodel) if file.endswith('h5')][0]
    
    with open(os.path.join(path_to_kmodel, model_json)) as f:
        model_json_string = f.read()
        
    model = model_from_json(model_json_string)
    model.load_weights(os.path.join(path_to_kmodel, model_weights))
    K.set_learning_phase(0)
    for i in range(3):
        layer = model.layers[-(i+1)]
        print(layer.name)
        print(layer.output)
        # print('11')
        print(layer.output.op.name)