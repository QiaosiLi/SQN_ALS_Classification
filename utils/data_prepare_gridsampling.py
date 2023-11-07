from colorsys import rgb_to_hls
from email.headerregistry import HeaderRegistry
from os.path import join, exists, dirname, abspath
from tkinter.ttk import LabeledScale
from wsgiref import headers
from xml.sax.handler import feature_string_interning
from xml.sax.xmlreader import AttributesImpl
import numpy as np
import glob, sys
import laspy

BASE_DIR = dirname(abspath(__file__))
ROOT_DIR = dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
from tool import DataProcessing
from tool import Plot



if __name__ == '__main__':

    dataset_name = 'Glasgow'  # todo change dataset name here
    # data_type = 'training' # or 'validation' or 'prediction'
    data_type = 'validation'   # todo specify if it is a training,validation,or prediction dataset here
    grid_size = 0.32
    total_class = 1

    DP = DataProcessing(dataset_name, grid_size)
    # nums_class = np.zeros(total_class, dtype=np.int32)

    for pc_path in glob.glob(join(DP.dataset_path, '*.las')):
        cloud_name = pc_path.split('/')[-1][:-4]
        print(cloud_name)

        # check if it has already calculated
        if exists(join(DP.sub_pc_folder, cloud_name + '_KDTree.pkl')):
            continue

        data = laspy.read(pc_path)
        xyz = data.xyz
        xyz = (xyz - np.amin(xyz, axis=0)).astype(np.float32)  # normalize
        # print(xyz) #todo
        intensity = (data.intensity).astype(np.float32)
        numberofreturns = np.array((data.number_of_returns)).astype(np.float32)
        returnnumber = np.array((data.return_number)).astype(np.float32)
        attributes = np.vstack((intensity, numberofreturns, returnnumber)).T
        if data_type == 'validation':
            labels = np.array(data.class_ann).astype(np.int32)
            print('using las.class_ann as labels')
        else:
            labels = np.array(data.classification).astype(np.int32)
            print('using las.classification as labels')

        print('summary of labels: {}'.format(np.unique(labels, return_counts=True)))

        DP.save_las2ply(cloud_name, xyz, attributes, labels=labels, grid_size=grid_size)

    print('Sampling finished')

