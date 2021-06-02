import os
import tensorflow as tf

from csv import writer
from time import time, strftime, gmtime

def reduce_mean_all(tensor_list):
    return [tf.reduce_mean(t) for t in tensor_list]        
    
def reduce_max_all(tensor_list):
    return [tf.reduce_max(t) for t in tensor_list]
        
def gpu_to_numpy(args:list):
    return [a.numpy() for a in args if tf.is_tensor(a)]
    
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

def append_to_results(ex_time:str, meta_args, loss_error:float, val_error:float):
    # Open file in append mode
    if not os.path.exists("experiments"):
        os.makedirs("experiments")
    
    args = [getattr(meta_args, arg) for arg in vars(meta_args)]

    with open("experiments/results.csv", 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow([strftime('%d.%m. %H:%M:%S', gmtime(time())), ex_time]+args+[loss_error, val_error])

