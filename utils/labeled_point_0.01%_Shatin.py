import glob
from os.path import join, exists, dirname, abspath, basename
import numpy as np
from helper_ply import write_ply, read_ply
import shutil

sub_ply_path = '/data/Shatin/input_0.320'
out_ply_path = '/data/Shatin/training/input'
ply_list = glob.glob(join(sub_ply_path,'*.ply'))
r = 0.001
print('randomly select {}% of labels for training '.format(r*100))
for ply_file in ply_list:
    ply_name = basename(ply_file)
    out_ply_file = join(out_ply_path, ply_name)
    if exists(out_ply_file):
        continue
    print(ply_name)
    data = read_ply(ply_file)
    sub_labels =  data['class']
    # print(data)
    print(np.unique(sub_labels, return_counts= True))
    new_labels = np.zeros_like(sub_labels, dtype=np.int32)
    num_pts = len(sub_labels)
    num_with_anno = max(int(num_pts * r), 1)
    valid_idx = np.where(sub_labels)[0]
    idx_with_anno = np.random.choice(valid_idx, num_with_anno, replace=False)
    new_labels[idx_with_anno] = sub_labels[idx_with_anno]
    xyz = np.vstack((data['x'], data['y'], data['z'])).T
    attributes = np.vstack((data['intensity'], data['numberofreturn'], data['returnnumber'])).T
    print(np.unique(new_labels, return_counts=True))
    write_ply(out_ply_file, (xyz, attributes, new_labels), ['x', 'y', 'z', 'intensity', 'numberofreturn', 'returnnumber', 'class'])

print('done')

# copy tree and project
print('copy kdtree and proj file to training folder:')
pkl_list = glob.glob(join(sub_ply_path,'*.pkl'))
for pkl_file in pkl_list:
    shutil.copy2(pkl_file, out_ply_path)
print('done')
