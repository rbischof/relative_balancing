import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from csv import writer
from time import time, strftime, gmtime

def reduce_mean_all(tensor_list):
    return [tf.reduce_mean(t) for t in tensor_list]        
    
def reduce_max_all(tensor_list):
    return [tf.reduce_max(t) for t in tensor_list]
        
def gpu_to_numpy(args:list):
    return [a.numpy() for a in args if tf.is_tensor(a) or isinstance(a, tf.Variable)]
    
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

def append_to_results(ex_time:str, meta_args, train_error:float, val_error:float):
    # Open file in append mode
    if not os.path.exists("experiments"):
        os.makedirs("experiments")
    
    args = [getattr(meta_args, arg) for arg in vars(meta_args)]

    if tf.is_tensor(train_error):
        train_error = train_error.numpy()
    if tf.is_tensor(val_error):
        val_error = val_error.numpy()

    with open("experiments/results.csv", 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow([strftime('%d.%m. %H:%M:%S', gmtime(time())), ex_time]+args+[train_error, val_error])

def show_image(img:np.array, path:str=None, extent:list=[0, 1, 0, 1], format='%.2f', x_label='x', y_label='y'):
    plt.rc('font', size=28) #controls default text size
    plt.rc('axes', labelsize=20)
    plt.rc('xtick', labelsize=20) #fontsize of the x tick labels
    plt.rc('ytick', labelsize=20) #fontsize of the y tick labels
    fig = plt.figure(figsize=(6, 4.5), dpi=100)
    ims = plt.imshow(img, cmap='plasma')
    cb = fig.colorbar(ims, format=format)
    cb.ax.tick_params(labelsize=20)
    plt.xticks([0, img.shape[1] // 2, img.shape[1]], [extent[0], extent[0]+(extent[1]-extent[0])/2, extent[1]])
    plt.yticks([0, img.shape[0] // 2, img.shape[0]], [extent[3], extent[2]+(extent[3]-extent[2])/2, extent[2]])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout(pad=0)
    if path is not None:
        plt.savefig(path, dpi=100)
    plt.show()
