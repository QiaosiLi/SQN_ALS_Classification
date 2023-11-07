import os
import pickle
from os.path import join, exists, dirname, abspath
import numpy as np
import glob, sys
import pandas as pd
from helper_ply import write_ply, read_ply
from sklearn.neighbors import KDTree
import shutil

def read_txts(folder_path, names, label):
    sub_folder = join(folder_path, names)
    txt_list = glob.glob(join(sub_folder, '*.txt'))
    if len(txt_list) != 0:
        data = [pd.read_csv(txt, header=None, delim_whitespace=True, dtype=np.float16) for txt in txt_list]
        data_df = pd.concat(data)
        data_np = data_df.values
        labels_np = np.full(data_np.shape[0], label)
        print('class:{0}, label:{1}, num of pts:{2}'.format(names,label, len(data_np)))
        return [data_np, labels_np]


if __name__ == '__main__':

    label_to_names = {
                      1: 'Unclassified',
                      2: 'Ground',
                      3: 'Vegetation',
                      4: 'Building'}

    dataset_path = '/data/Glasgow/training'

    labels_folder =join(dataset_path, 'labels')
    # if not os.path.exists(labels_folder):
    #     os.makedirs(labels_folder)
    cloud_folder = glob.glob(labels_folder + '/*/', recursive = True)
    for pc_path in cloud_folder:
        cloud_name = pc_path.split('/')[-2]
        print('pc name: {}'.format(cloud_name))

        data = []
        labels = []
        for l, n in label_to_names.items():
            read = read_txts(pc_path, n, l)
            if read is None:
                continue
            data.append(read[0])
            labels.append(read[1])

        ann_data_np = np.concatenate(data)
        ann_labels_np = np.concatenate(labels)
        # print(len(ann_data_np))
        print('num of pt in {}: {}'.format(cloud_name, len(ann_data_np)))

        ann_data = ann_data_np.astype(np.float32)
        ann_labels = ann_labels_np.astype(np.int32)
        ann_labels = np.reshape(ann_labels,(-1,1))
        ann_data = np.hstack((ann_data, ann_labels))
        npy_file = join(pc_path, cloud_name+'.npy')
        np.save(npy_file, ann_data)

        gridsampling_path = '/data/Glasgow'
        gridsampling_folder = join(gridsampling_path, 'input_0.320')
        sub_ply_file = join(gridsampling_folder, cloud_name + '.ply')
        kd_tree_file = join(gridsampling_folder, cloud_name+'_KDTree.pkl')
        proj_file = join(gridsampling_folder, cloud_name+'_proj.pkl')
        ann_npy_file = npy_file
        labeled_data_folder = join(dataset_path, 'labeled_data')
        if not os.path.exists(labeled_data_folder):
            os.makedirs(labeled_data_folder)
        labeled_file = join(labeled_data_folder, cloud_name + '.ply')

        sub_data = read_ply(sub_ply_file)
        ann_data = np.load(ann_npy_file)
        with open(kd_tree_file, 'rb') as f:
            search_tree = pickle.load(f)
        ann_xyz = ann_data[:, 0:3]
        print(np.unique(ann_data[:, -1], return_counts=True))
        proj_idx = np.squeeze(search_tree.query(ann_xyz, return_distance=False)).astype(np.int32)
        new_labels = np.zeros_like(sub_data['class'], dtype=np.int32)
        new_labels[proj_idx] = ann_data[:, -1]
        print(np.unique(new_labels, return_counts=True))
        sub_xyz = np.stack((sub_data['x'], sub_data['y'], sub_data['z']), 1)
        sub_attributes = np.stack((sub_data['intensity'], sub_data['numberofreturn'], sub_data['returnnumber']), 1)
        write_ply(labeled_file,
                  (sub_xyz, sub_attributes, new_labels),
                  ['x', 'y', 'z', 'intensity', 'numberofreturn', 'returnnumber', 'class'])

        # copy data to input folder
        input_folder = join(dataset_path, 'input')
        if not os.path.exists(labeled_data_folder):
            os.makedirs(labeled_data_folder)
        shutil.copy2(labeled_file, input_folder)
        shutil.copy2(kd_tree_file, input_folder)
        shutil.copy2(proj_file, input_folder)

print('labeling finished')