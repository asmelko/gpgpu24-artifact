#!/usr/bin/python3

import sys
import os
import numpy as np
import ntpath

if len(sys.argv) != 2:
    print('ERROR. Expecting one argument, a topk and projection results file.')
    exit()

rootdir = sys.argv[1]
output_data_dir = os.path.join(rootdir, 'processed_data')

to_exclude_dirs = [output_data_dir, os.path.join(rootdir, 'projection-badk')]

save_complete_data = True

class Object(object):
    pass

try:
    os.mkdir(output_data_dir)
except:
    pass

def preprocess_data_and_create_means():
    full_dict = {}

    for subdir, dirs, files in os.walk(rootdir):
        if subdir in to_exclude_dirs:
            continue

        for file in files:
            with open(os.path.join(subdir, file), "r") as a_file:
                for line in a_file:
                    stripped_line = line.strip()
                    split_line = stripped_line.split(';')

                    (type, algo, N, dim, gridSize, k, cudaBlockSize, cudaSharedMemorySize, groupsPerBlock, itemsPerBlock, itemsPerThread, initTime, kernelTime, empty) = split_line

                    key = ';'.join([type, algo, N, dim, gridSize, k, cudaBlockSize, cudaSharedMemorySize, groupsPerBlock, itemsPerBlock, itemsPerThread])
                    
                    if key not in full_dict:
                        full_dict[key] = []
                    full_dict[key].append(kernelTime)

    for key, value in full_dict.items():
        repeats = len(value)
        arr = np.array(value).astype(np.float32)
        full_dict[key] = [np.mean(arr), np.std(arr)]

    return full_dict

def process_entries(data, key_creator, value_creator, output):
    min_dict = {}

    for key, times in data.items():
        key = key.split(';')
        time, std = times
        (type, algo, N, dim, gridSize, k, cudaBlockSize, cudaSharedMemorySize, groupsPerBlock, itemsPerBlock, itemsPerThread) = key

        new_key = key_creator(*key)
        new_value = value_creator(*key)

        if new_key not in min_dict:
            min_dict[new_key] = Object()
            min_dict[new_key].config = new_value
            min_dict[new_key].time = time
        elif min_dict[new_key].time > time:
            min_dict[new_key].config = new_value
            min_dict[new_key].time = time

    f = open(output, "w")
    for key in min_dict:
        f.write(';'.join([key, min_dict[key].config,str(min_dict[key].time)]) + '\n')
    f.close()


data = preprocess_data_and_create_means()

if save_complete_data:
    f = open(os.path.join(output_data_dir, 'complete_data'), "w")
    for key, times in data.items():
        f.write(';'.join([key, str(times[0]), str(times[1])]) + '\n')
    f.close()

def create_algo_config(type, algo, N, dim, gridSize, k, cudaBlockSize, cudaSharedMemorySize, groupsPerBlock, itemsPerBlock, itemsPerThread):
    return ';'.join([cudaBlockSize, cudaSharedMemorySize, groupsPerBlock, itemsPerBlock, itemsPerThread])

def create_algo_key(type, algo, N, dim, gridSize, k, cudaBlockSize, cudaSharedMemorySize, groupsPerBlock, itemsPerBlock, itemsPerThread):
    return ';'.join([type, algo, N, dim, gridSize, k])

def create_type_config(type, algo, N, dim, gridSize, k, cudaBlockSize, cudaSharedMemorySize, groupsPerBlock, itemsPerBlock, itemsPerThread):
    return ';'.join([algo, cudaBlockSize, cudaSharedMemorySize, groupsPerBlock, itemsPerBlock, itemsPerThread])

def create_type_key(type, algo, N, dim, gridSize, k, cudaBlockSize, cudaSharedMemorySize, groupsPerBlock, itemsPerBlock, itemsPerThread):
    return ';'.join([type, N, dim, gridSize, k])

process_entries(data, create_algo_key, create_algo_config, os.path.join(output_data_dir, 'best_alg_configs'))
process_entries(data, create_type_key, create_type_config, os.path.join(output_data_dir, 'overall_best_algs'))
