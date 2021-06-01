import os
import tensorflow as tf

def reduce_mean_all(tensor_list):
    return [tf.reduce_mean(t) for t in tensor_list]        
    
def reduce_max_all(tensor_list):
    return [tf.reduce_max(t) for t in tensor_list]
        
def gpu_to_numpy(args:list):
    return [a.numpy() for a in args]
    
def create_directory(path:str):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        ix = 1
        if path[-1] == '/':
            path = path[:-1]
        alternative_path = path + '_' + str(ix) + '/'
        while os.path.exists(alternative_path):
            ix += 1
            alternative_path = path + '_' + str(ix) + '/'
        path = alternative_path
        os.makedirs(path)
    return path