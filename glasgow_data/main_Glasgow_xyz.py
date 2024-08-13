from os.path import join, exists, dirname, abspath
from SQN import Network
from tester_Glasgow import ModelTester
from helper_ply import read_ply
from tool import ConfigGlasgow as cfg
from tool import DataProcessing as DP
from tool import Plot
import tensorflow as tf
import numpy as np
import time, pickle, argparse, glob, os, shutil
from pathlib import Path


class Glasgow:
    def __init__(self):
        self.name = 'Glasgow'
        root_path= '/data'
        self.path = join(root_path, self.name)
        if not Path(root_path).exists():
            raise ValueError('Path does not exist')
        self.label_to_names = {0: 'Unclassified',
                               1: 'Others',
                               2: 'Ground',
                               3: 'Vegetation',
                               4: 'Buildings'}
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.array([0])

        self.use_val = True  # whether use validation set or not
        self.val_split = 'val_data'
        self.train_files_path = join(self.path, 'training', 'input')
        self.val_files_path = join(self.path, 'validation', 'input')
        # pre_files_path = join(self.path, 'test_data')
        self.train_files = np.sort(glob.glob(join(self.train_files_path,'*.ply')))
        self.val_files = np.sort(glob.glob(join(self.val_files_path,'*.ply')))
        # self.all_files = np.sort(glob.glob(join(self.path, 'original_ply', '*.ply'))) # todo


        self.num_with_anno_per_batch = cfg.num_points
        self.num_per_class = np.zeros(self.num_classes)
        self.val_proj = []
        self.val_labels = []
        self.possibility = {}
        self.min_possibility = {}
        self.input_trees = {'training': [], 'validation': []}
        self.input_colors = {'training': [], 'validation': []}
        self.input_labels = {'training': [], 'validation': []}
        self.input_names = {'training': [], 'validation': []}


        self.load_sub_sampled_clouds()
        for ignore_label in self.ignored_labels:
            self.num_per_class = np.delete(self.num_per_class, ignore_label)


    def load_sub_sampled_clouds(self):
        # load train pc
        train_path = self.train_files_path
        val_path = self.val_files_path
        for i, file_path in enumerate(self.train_files):
            t0 = time.time()
            cloud_name = file_path.split('/')[-1][:-4]
            print('Load_training_pc:' + cloud_name)
            kd_tree_file = join(train_path, '{:s}_KDTree.pkl'.format(cloud_name))
            ann_ply_file = join(train_path, '{:s}.ply'.format(cloud_name))

            # read ply with data
            data = read_ply(ann_ply_file)
            ann_attributes = np.vstack((data['intensity'], data['numberofreturn'], data['returnnumber'])).T
            ann_attributes = np.zeros_like(ann_attributes)  # attributes are replaced by zeros
            ann_labels = data['class']
            # data.dtype.names  ('x', 'y', 'z', 'intensity', 'numberofreturn', 'returnnumber', 'class')
            self.num_per_class += DP.get_num_class_from_label(ann_labels, self.num_classes)

            with open(kd_tree_file, 'rb') as f:
                search_tree = pickle.load(f)

            self.input_trees['training'] += [search_tree]
            self.input_colors['training'] += [ann_attributes]
            self.input_labels['training'] += [ann_labels]
            self.input_names['training'] += [cloud_name]

            # size = ann_attributes.shape[0] * 4 * 7
            size = ann_labels.shape[0] * 4 * 7
            print('{:s} {:.1f} MB loaded in {:.1f}s'.format(kd_tree_file.split('/')[-1], size * 1e-6, time.time() - t0))

        print('\nPreparing reprojected indices for testing')

        # Get validation and test reprojected indices

        for i, file_path in enumerate(self.val_files):
            t0 = time.time()
            cloud_name = file_path.split('/')[-1][:-4]
            print('Load_validation_pc:' + cloud_name)

            kd_tree_file = join(val_path, '{:s}_KDTree.pkl'.format(cloud_name))
            val_ply_file = join(val_path, '{:s}.ply'.format(cloud_name))

            # read ply with data
            data = read_ply(val_ply_file)
            sub_attributes = np.vstack((data['intensity'], data['numberofreturn'], data['returnnumber'])).T
            sub_attributes = np.zeros_like(sub_attributes)
            sub_labels = data['class']  # data.dtype.names  ('x', 'y', 'z', 'intensity', 'numberofreturn', 'returnnumber', 'class')

            with open(kd_tree_file, 'rb') as f:
                search_tree = pickle.load(f)

            self.input_trees['validation'] += [search_tree]
            self.input_colors['validation'] += [sub_attributes]
            self.input_labels['validation'] += [sub_labels]
            self.input_names['validation'] += [cloud_name]

            proj_file = join(val_path, '{:s}_proj.pkl'.format(cloud_name))
            with open(proj_file, 'rb') as f:
                proj_idx, labels = pickle.load(f)
            self.val_proj += [proj_idx]
            self.val_labels += [labels]
            print('{:s} done in {:.1f}s'.format(cloud_name, time.time() - t0))


    def get_batch_gen(self, split):
        if split == 'training':
            num_per_epoch = cfg.train_steps * cfg.batch_size
        elif split == 'validation':
            num_per_epoch = cfg.val_steps * cfg.val_batch_size

        # Reset possibility
        self.possibility[split] = []
        self.min_possibility[split] = []
        for i, tree in enumerate(self.input_colors[split]):
            # for i, tree in enumerate(self.input_labels[split]):
            self.possibility[split] += [np.random.rand(tree.data.shape[0]) * 1e-3]
            self.min_possibility[split] += [float(np.min(self.possibility[split][-1]))]

        def spatially_regular_gen():
            # Generator loop
            for i in range(num_per_epoch):  # num_per_epoch

                # Choose a random cloud
                cloud_idx = int(np.argmin(self.min_possibility[split]))

                # choose the point with the minimum of possibility as query point
                point_ind = np.argmin(self.possibility[split][cloud_idx])

                # Get points from tree structure
                points = np.array(self.input_trees[split][cloud_idx].data, copy=False)

                # Center point of input region
                center_point = points[point_ind, :].reshape(1, -1)

                # Add noise to the center point
                noise = np.random.normal(scale=cfg.noise_init / 10, size=center_point.shape)
                pick_point = center_point + noise.astype(center_point.dtype)

                # print('point shape is here')
                # print(points.shape)

                if len(points) < cfg.num_points:
                    queried_idx = self.input_trees[split][cloud_idx].query(pick_point, k=len(points))[1][0]
                else:
                    queried_idx = self.input_trees[split][cloud_idx].query(pick_point, k=cfg.num_points)[1][0]

                queried_idx = DP.shuffle_idx(queried_idx)
                # Collect points and colors
                queried_pc_xyz = points[queried_idx]
                queried_pc_xyz = queried_pc_xyz - pick_point
                queried_pc_colors = self.input_colors[split][cloud_idx][queried_idx]
                queried_pc_labels = self.input_labels[split][cloud_idx][queried_idx]

                dists = np.sum(np.square((points[queried_idx] - pick_point).astype(np.float32)), axis=1)
                delta = np.square(1 - dists / np.max(dists))
                self.possibility[split][cloud_idx][queried_idx] += delta
                self.min_possibility[split][cloud_idx] = float(np.min(self.possibility[split][cloud_idx]))

                if len(points) < cfg.num_points:
                    queried_pc_xyz, queried_pc_colors, queried_idx, queried_pc_labels = \
                        DP.data_aug(queried_pc_xyz, queried_pc_colors, queried_pc_labels, queried_idx, cfg.num_points)

                if split == 'training':
                    unique_label_value = np.unique(queried_pc_labels)
                    if len(unique_label_value) <= 1:
                        i -= 1
                        continue
                    else:
                        # ================================================================== #
                        #            Keep the same number of labeled points per batch        #
                        # ================================================================== #
                        idx_with_anno = np.where(queried_pc_labels != self.ignored_labels[0])[0]
                        num_with_anno = len(idx_with_anno)
                        if num_with_anno > self.num_with_anno_per_batch:
                            idx_with_anno = np.random.choice(idx_with_anno, self.num_with_anno_per_batch, replace=False)
                        elif num_with_anno < self.num_with_anno_per_batch:
                            dup_idx = np.random.choice(idx_with_anno, self.num_with_anno_per_batch - len(idx_with_anno))
                            idx_with_anno = np.concatenate([idx_with_anno, dup_idx], axis=0)
                        xyz_with_anno = queried_pc_xyz[idx_with_anno]
                        labels_with_anno = queried_pc_labels[idx_with_anno]
                else:
                    xyz_with_anno = queried_pc_xyz
                    labels_with_anno = queried_pc_labels

                if True:
                    yield (queried_pc_xyz.astype(np.float32),
                           queried_pc_colors.astype(np.float32),
                           queried_pc_labels,
                           queried_idx.astype(np.int32),
                           np.array([cloud_idx], dtype=np.int32),
                           xyz_with_anno.astype(np.float32),
                           labels_with_anno.astype(np.int32))

        gen_func = spatially_regular_gen
        gen_types = (tf.float32, tf.float32, tf.int32, tf.int32, tf.int32, tf.float32, tf.int32)
        gen_shapes = ([None, 3], [None, 3], [None], [None], [None], [None, 3], [None])
        return gen_func, gen_types, gen_shapes

    @staticmethod
    def get_tf_mapping2():

        def tf_map(batch_xyz, batch_features, batch_labels, batch_pc_idx, batch_cloud_idx, batch_xyz_anno,
                   batch_label_anno):
            batch_features = tf.concat([batch_xyz, batch_features], axis=-1)
            input_points = []
            input_neighbors = []
            input_pools = []
            input_up_samples = []

            for i in range(cfg.num_layers):
                neighbour_idx = tf.py_func(DP.knn_search, [batch_xyz, batch_xyz, cfg.k_n], tf.int32)
                sub_points = batch_xyz[:, :tf.shape(batch_xyz)[1] // cfg.sub_sampling_ratio[i], :]
                pool_i = neighbour_idx[:, :tf.shape(batch_xyz)[1] // cfg.sub_sampling_ratio[i], :]
                up_i = tf.py_func(DP.knn_search, [sub_points, batch_xyz, 1], tf.int32)
                input_points.append(batch_xyz)
                input_neighbors.append(neighbour_idx)
                input_pools.append(pool_i)
                input_up_samples.append(up_i)
                batch_xyz = sub_points

            input_list = input_points + input_neighbors + input_pools + input_up_samples
            input_list += [batch_features, batch_labels, batch_pc_idx, batch_cloud_idx, batch_xyz_anno,
                           batch_label_anno]

            return input_list

        return tf_map

    def init_input_pipeline(self):
        print('Initiating input pipelines')
        cfg.ignored_label_inds = [self.label_to_idx[ign_label] for ign_label in self.ignored_labels]
        gen_function, gen_types, gen_shapes = self.get_batch_gen('training')
        gen_function_val, _, _ = self.get_batch_gen('validation')
        self.train_data = tf.data.Dataset.from_generator(gen_function, gen_types, gen_shapes)
        self.val_data = tf.data.Dataset.from_generator(gen_function_val, gen_types, gen_shapes)

        self.batch_train_data = self.train_data.batch(cfg.batch_size)
        self.batch_val_data = self.val_data.batch(cfg.val_batch_size)
        map_func = self.get_tf_mapping2()

        self.batch_train_data = self.batch_train_data.map(map_func=map_func)
        self.batch_val_data = self.batch_val_data.map(map_func=map_func)

        self.batch_train_data = self.batch_train_data.prefetch(cfg.batch_size)
        self.batch_val_data = self.batch_val_data.prefetch(cfg.val_batch_size)

        iter = tf.data.Iterator.from_structure(self.batch_train_data.output_types, self.batch_train_data.output_shapes)
        self.flat_inputs = iter.get_next()
        self.train_init_op = iter.make_initializer(self.batch_train_data)
        self.val_init_op = iter.make_initializer(self.batch_val_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='the number of GPUs to use [default: 0]')
    # parser.add_argument('--test_area', type=int, default=2, help='Which area to use for test, option: 1-6 [default: 5]')
    parser.add_argument('--mode', type=str, default='train', help='options: train, test, vis')
    # parser.add_argument('--labeled_point', type=str, default='0.1%', help='0.1%/1%/10%/100%')
    # parser.add_argument('--gen_pseudo', default=False, action='store_true', help='generate pseudo labels or not')
    # parser.add_argument('--retrain', default=False, action='store_true', help='Re-training with pseudo labels or not')
    FLAGS = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    Mode = FLAGS.mode

    print('Settings:')
    print('Mode:', FLAGS.mode)
    # print('Labeled_point', FLAGS.labeled_point)
    # print('gen_pseudo', FLAGS.gen_pseudo)
    # print('retrain', FLAGS.retrain)

    shutil.rmtree('__pycache__') if exists('__pycache__') else None
    if Mode == 'train':
        # shutil.rmtree('results') if exists('results') else None
        shutil.rmtree('checked_Script/train_log') if exists('checked_Script/train_log') else None
        for f in os.listdir(dirname(abspath(__file__))):
            if f.startswith('log_'):
                os.remove(f)


    dataset = Glasgow()
    dataset.init_input_pipeline()

    if Mode == 'train':
        # model = Network(dataset, cfg, FLAGS.retrain)
        model = Network(dataset, cfg)
        model.train(dataset)
    elif Mode == 'test':
        cfg.saving = False
        model = Network(dataset, cfg)
        chosen_snapshot = -1
        logs = np.sort([os.path.join('checked_Script/results', f) for f in os.listdir('checked_Script/results') if f.startswith('Log')])
        chosen_folder = logs[-1]
        snap_path = join(chosen_folder, 'snapshots')
        # snap_path = '/home/qll3h/ql_project/SQN_ALS_Classification/models/glasgow_xyz/snapshots' # xyz only,  todo: replace the path of pretrain model here
        snap_steps = [int(f[:-5].split('-')[-1]) for f in os.listdir(snap_path) if f[-5:] == '.meta']
        chosen_step = np.sort(snap_steps)[-1]
        chosen_snap = os.path.join(snap_path, 'snap-{:d}'.format(chosen_step))
        tester = ModelTester(model, dataset, restore_snap=chosen_snap)
        # tester.test(model, dataset, FLAGS.gen_pseudo)
        tester.test(model, dataset)
        shutil.rmtree('checked_Script/train_log') if exists('checked_Script/train_log') else None

    else:

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(dataset.train_init_op)
            while True:
                data_list = sess.run(dataset.flat_inputs)
                xyz = data_list[0]
                sub_xyz = data_list[1]
                label = data_list[21]
                Plot.draw_pc_sem_ins(xyz[0, :, :], label[0, :])
                Plot.draw_pc_sem_ins(sub_xyz[0, :, :], label[0, 0:np.shape(sub_xyz)[1]])
