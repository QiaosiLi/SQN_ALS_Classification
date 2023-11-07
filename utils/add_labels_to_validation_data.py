import os
from os.path import basename, join, isdir
import numpy as np
from sklearn.neighbors import KDTree
import laspy
import pickle
import glob
import pandas as pd

def read_txts(folder_path, names, label):
    sub_folder = join(folder_path, names)
    txt_list = glob.glob(join(sub_folder, '*.txt'))
    if len(txt_list) != 0:
        data = [pd.read_csv(txt, header=None, delim_whitespace=True, dtype=np.float64) for txt in txt_list]
        data_df = pd.concat(data)
        data_np = data_df.values
        labels_np = np.full(data_np.shape[0], label)
        # print(len(data_np))
        print('class:{0}, label:{1}, num of pts:{2}'.format(names,label, len(data_np)))
        return [data_np, labels_np]

if __name__ == '__main__':
    # change your data folder path here, use tile name as folder name
    dataset_path = '/data/Glasgow/validation'
    labels_folder =join(dataset_path, 'labels')
    cloud_folder = glob.glob(labels_folder + '/*/', recursive = True)
    for pc_path in cloud_folder:
        cloud_name = pc_path.split('/')[-2]
        print('pc name: {}'.format(cloud_name))

        las_file = join(dataset_path, 'original_las', cloud_name+'.las')
        las_data = laspy.read(las_file)
        xyz = las_data.xyz
        kd_tree_file = join(dataset_path, 'original_las', cloud_name + '_KDTree.pkl')
        # with open(kd_tree_file, 'wb') as f:
        #     pickle.dump(xyz_search_tree, f)

        if not os.path.exists(kd_tree_file):
            xyz_search_tree = KDTree(xyz, leaf_size=50)
            with open(kd_tree_file, 'wb') as f:
                pickle.dump(xyz_search_tree, f)
        else:
            with open(kd_tree_file, 'rb') as f:
                xyz_search_tree = pickle.load(f)

        ## read txt
        label_to_names = {
            1: 'Unclassified',
            2: 'Ground',
            3: 'Vegetation',
            4: 'Building'}
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
        print('num of pt in txt: {}'.format(len(ann_data_np)))
        print('num of pt in las: {}'.format(len(las_data)))

        ann_data = ann_data_np.astype(np.float64)
        ann_labels = np.reshape(ann_labels_np, (-1, 1))
        ann_data = np.hstack((ann_data, ann_labels))
        npy_file = join(pc_path, cloud_name + '.npy')
        np.save(npy_file, ann_data)

        if len(ann_data_np) != len(las_data):
            print('The number of annotated pts and las pts is not equal')
            ## Find the missed points
            # ann_xyz = ann_data[:, 0:3]
            # proj_idx = np.squeeze(search_tree.query(ann_xyz, return_distance=False)).astype(np.int32)
            # idx = np.arange(len(las_data))
            # dif_idx = np.setdiff1d(idx, proj_idx)
            # mis_points =las_data.points[dif_idx].copy()
            # mis_las_data = laspy.LasData(las_data.header)
            # mis_las_data.points = mis_points
            # mis_las_data.write('/home/MissingPts.las')

        else:
            ann_xyz = ann_data[:, 0:3]
            print('summary of annotated labels:{}'.format(np.unique(ann_data[:, -1], return_counts=True)))
            proj_idx = np.squeeze(xyz_search_tree.query(ann_xyz, return_distance=False)).astype(np.int32)
            new_labels = np.zeros_like(np.array(las_data.classification), dtype=np.int32)
            new_labels[proj_idx] = ann_data[:, -1]
            print('summary of new labels: {}'.format(np.unique(new_labels, return_counts=True)))
            las_data.add_extra_dim(laspy.ExtraBytesParams(name='class_ann', type=np.int32))
            las_data.points['class_ann'] = new_labels
            out_las_file = join(dataset_path, 'labeled_data', cloud_name + '_Ann.las')
            las_data.write(out_las_file)

    print('labeled las data are saved.')




