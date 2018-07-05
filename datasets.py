import utils
from coco import CocoDataset
import numpy as np
from collections import namedtuple
import os
import os.path as osp
from cityscape_labels import labels as CITYSCAPE_LABELS
from bdd_labels import labels as BDD_LABELS
from skimage.io import imread

DatasetTuple = namedtuple('DatasetTuple', ['dataset', 'starts_from'])

CITYSCAPE_TRAIN_NAMES = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign',
                         'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
                         'motorcycle', 'bicycle']

ACTIVE_COCO_IDS = [1, 2, 3, 4, 6, 7, 8, 10, 13]


class MyCocoDataset(CocoDataset):
    def __init__(self, dataset_dir, subset='minival', class_map=None):
        super(MyCocoDataset, self).__init__()
        self.coco = self.load_coco(dataset_dir, subset, return_coco=True, class_map=class_map,
                                   class_ids=ACTIVE_COCO_IDS)
        self.prepare()


class BddDataset(utils.Dataset):
    def __init__(self, dataset_dir, subset, class_map=None):
        super(BddDataset, self).__init__()
        self._image_dir = osp.join(dataset_dir, 'images', subset)
        self._labels_dir = osp.join(dataset_dir, 'labels', subset)
        self._class_map = class_map
        self.load_dataset()
        self.prepare()

    def load_dataset(self):
        # call add_class to add class ids.
        # call add_image to add image info + annotations
        class_ids = set()
        for label in CITYSCAPE_LABELS:
            if label.trainId not in class_ids:
                self.add_class('bdd', label.trainId, label.name)
                class_ids.add(label.trainId)

        for image_path in os.listdir(self._image_dir):
            self.add_image('bdd', osp.splitext(image_path)[0], osp.join(self._image_dir, image_path))

    def load_mask(self, image_id):
        label_path = osp.join(self._labels_dir, self.image_info[image_id]['id'] + '_train_id.png')
        label = imread(label_path)
        print(label.shape)
        if label.ndim > 2:
            seg = label[:, :, 0]
            instance = label[:, :, 1]
            unique_class_ids = np.unique(seg)
            unique_class_ids = unique_class_ids[unique_class_ids != 255]
            class_ids = []
            instance_masks = []
            for class_id in unique_class_ids:
                class_instances = np.unique(instance[seg == class_id])
                for instance_id in class_instances:
                    bin_mask = np.zeros_like(seg)
                    bin_mask[seg == class_id & instance == instance_id] = 1
                    instance_masks.append(bin_mask)
                    class_ids.append(class_id)

            mask = np.stack(instance_masks, axis=2)

        else:
            seg = label
            class_ids = np.unique(seg)
            class_ids = class_ids[class_ids != 255]
            mask = np.zeros((*label.shape, len(class_ids)), dtype=label.dtype)
            for i, class_id in enumerate(class_ids):
                mask[label == class_id, i] = 1

        return mask, class_ids


class CityScapeDataset(utils.Dataset):
    def __init__(self, dataset_dir, subset, class_map=None):
        super(CityScapeDataset, self).__init__()
        self._image_dir = osp.join(dataset_dir, 'leftImg8bit', subset)
        self._labels_dir = osp.join(dataset_dir, 'gtFine', subset)
        self._class_map = class_map
        self.load_dataset()
        self.prepare()

    def load_dataset(self):
        for label in CITYSCAPE_LABELS:
                self.add_class('bdd', label.id, label.name)

        for city in os.listdir(self._image_dir):
            for image_path in os.listdir(osp.join(self._image_dir, city)):
                self.add_image('bdd', '_'.join(image_path.split('_')[:-1]), osp.join(self._image_dir, city, image_path))

    def load_mask(self, image_id):
        id = self.image_info[image_id]['id']
        city = id.split('_')[0]
        label = imread(osp.join(self._labels_dir, city, id + '_gtFine_instanceIds.png'))
        instance_ids = np.unique(label)
        # don't create mask for BG
        instance_ids = instance_ids[instance_ids != 0]

        instance_masks = []
        class_ids = []
        for i, instance_id in enumerate(instance_ids):
            class_id = instance_id if instance_id < 1000 else instance_id // 1000
            if self._class_map is not None:
                class_id = self._class_map[class_id]
                if class_id == 255:
                    continue

            mask = np.zeros_like(label, dtype=np.bool)
            mask[label == instance_id] = 1
            instance_masks.append(mask)
            class_ids.append(class_id)

        mask = np.stack(instance_masks, axis=2)
        class_ids = np.array(class_ids)

        return mask, class_ids



# create a BddDataset object for training/eval dataset
# load the training/eval datasets
# call prepare for each
# call model.train_model


class CombinedDataset(utils.Dataset):

    def __init__(self):
        super(CombinedDataset, self).__init__()
        self.datasets = {}
        self.num_images = 0
        self.map_id = None  # should be a function that returns a tuple (source, source_id) from image id
        self._start_ids = None

    def add_dataset(self, source, dataset):
        self.datasets[source] = DatasetTuple(dataset, self.num_images)

        self.num_images += len(dataset.image_ids)

    def prepare(self):
        # similar to utils.prepare
        # must generate map_id
        start_ids = []
        for source in self.datasets:
            start_ids.append((self.datasets[source].starts_from, source))
        self._start_ids = sorted(start_ids, reverse=True)

        def map_id(id):
            if id > self.num_images or id < 0:
                raise KeyError

            for start_id, source in self._start_ids:
                if id >= start_id:
                    return source, id - start_id

        self.map_id = map_id
        # TODO: check for class_maps and craete new class_ids and class_names

    def load_image(self, image_id):
        source, source_id = self.map_id(image_id)
        return self.datasets[source].dataset.load_image(source_id)

    def load_mask(self, image_id):
        source, source_id = self.map_id(image_id)
        mask, class_ids = self.datasets[source].dataset.load_mask(source_id)
        return mask, class_ids

    @property
    def image_ids(self):
        return np.arange(self.num_images)
