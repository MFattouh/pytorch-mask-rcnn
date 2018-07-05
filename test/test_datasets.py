import unittest
from datasets import *
import numpy as np
import random
import matplotlib.pyplot as plt
import yaml
from demo import class_names as COCO_CLASS_NAMES
# VISUALIZE = False
VISUALIZE = True


def visualize_mask(image, mask, class_ids, class_names):
    plt.figure()
    plt.imshow(image)
    for i in range(mask.shape[-1]):
        plt.figure(frameon=False)
        plt.imshow(image, alpha=0.5)
        plt.imshow(mask[:, :, i], alpha=0.5)
        plt.title(class_names[class_ids[i]])

    plt.show()


class TestMyCoco(unittest.TestCase):
    def setUp(self):
        self.coco = MyCocoDataset('/home/fattouhm/datasets/coco')

    def test_load_image(self):
        image_id = random.sample(self.coco.image_ids.tolist(), 1)[0]
        image = self.coco.load_image(image_id)
        if VISUALIZE:
            plt.imshow(image)
            plt.show()

    def test_load_mask(self):
        image_id = random.sample(self.coco.image_ids.tolist(), 1)[0]
        image = self.coco.load_image(image_id)
        mask, class_ids = self.coco.load_mask(image_id)
        if VISUALIZE:
            visualize_mask(image, mask, class_ids, COCO_CLASS_NAMES)

    def test_class_mapping(self):
        del self.coco
        with open('id_mapping.yaml', 'r') as fp:
            class_map = yaml.load(fp)['COCO']
        self.coco = MyCocoDataset('/home/fattouhm/datasets/coco', class_map=class_map)
        image_id = random.sample(self.coco.image_ids.tolist(), 1)[0]
        image = self.coco.load_image(image_id)
        mask, class_ids = self.coco.load_mask(image_id)
        if VISUALIZE:
            visualize_mask(image, mask, class_ids, CITYSCAPE_TRAIN_NAMES)


class TestBdd(unittest.TestCase):
    def setUp(self):
        self.dataset = BddDataset('/home/fattouhm/datasets/bdd/bdd100k/seg', 'val')
        self.dataset.prepare()

    def test_load_image(self):
        image_id = random.sample(self.dataset.image_ids.tolist(), 1)[0]
        image = self.dataset.load_image(image_id)
        if VISUALIZE:
            plt.figure()
            plt.imshow(image)
            plt.show()

    def test_load_mask(self):
        image_id = random.sample(self.dataset.image_ids.tolist(), 1)[0]
        image = self.dataset.load_image(image_id)
        mask, class_ids = self.dataset.load_mask(image_id)
        print(mask.shape)
        print(class_ids)
        if VISUALIZE:
            visualize_mask(image, mask, class_ids, self.dataset.class_names)


class TestCityscape(unittest.TestCase):
    def setUp(self):
        self.dataset = CityScapeDataset('/home/fattouhm/datasets/cityscape/', 'val')
        self.dataset.prepare()

    def test_load_image(self):
        image_id = random.sample(self.dataset.image_ids.tolist(), 1)[0]
        image = self.dataset.load_image(image_id)
        if VISUALIZE:
            plt.figure()
            plt.imshow(image)
            plt.show()

    def test_load_mask(self):
        image_id = random.sample(self.dataset.image_ids.tolist(), 1)[0]
        image = self.dataset.load_image(image_id)
        mask, class_ids = self.dataset.load_mask(image_id)
        if VISUALIZE:
            visualize_mask(image, mask, class_ids, self.dataset.class_names)

    def test_class_mapping(self):
        del self.dataset
        with open('id_mapping.yaml') as fp:
            cityscape_map = yaml.load(fp)['CITYSCAPE']
        self.dataset = CityScapeDataset('/home/fattouhm/datasets/cityscape/', 'val', cityscape_map)
        image_id = random.sample(self.dataset.image_ids.tolist(), 1)[0]
        image = self.dataset.load_image(image_id)
        mask, class_ids = self.dataset.load_mask(image_id)
        if VISUALIZE:
            visualize_mask(image, mask, class_ids, CITYSCAPE_TRAIN_NAMES)


class TestCombined(unittest.TestCase):
    def setUp(self):
        self.coco = MyCocoDataset('/home/fattouhm/datasets/coco')
        self.combined = CombinedDataset()
        self.combined.add_dataset('coco', self.coco)

    def test_one_dataset(self):
        self.combined.prepare()
        self.assertEqual(self.combined.num_images, len(self.coco.image_ids), 'different number of images')
        image_id = random.sample(self.combined.image_ids.tolist(), 1)[0]
        combined_image = self.combined.load_image(image_id)
        coco_image = self.coco.load_image(image_id)
        np.testing.assert_equal(combined_image, coco_image, 'different images returned')

        mask, class_ids = self.combined.load_mask(image_id)
        if VISUALIZE:
            visualize_mask(combined_image, mask, class_ids, self.coco.class_names)

    def test_load_first(self):
        self.combined.prepare()
        image_id = 0
        combined_image = self.combined.load_image(image_id)
        coco_image = self.coco.load_image(image_id)
        np.testing.assert_equal(combined_image, coco_image, 'different images returned')

    def test_load_last(self):
        self.combined.prepare()
        image_id = self.coco.num_images - 1
        combined_image = self.combined.load_image(image_id)
        coco_image = self.coco.load_image(image_id)
        np.testing.assert_equal(combined_image, coco_image, 'different images returned')

    def test_id_mapping(self):
        id_map = dict(zip(self.coco.class_ids, (np.array(self.coco.class_ids) + 1).tolist()))
        coco1 = MyCocoDataset('/home/fattouhm/datasets/coco', class_map=id_map)
        self.combined.add_dataset('COCO1', coco1)
        self.combined.prepare()
        # sample from the second dataset
        coco_id = random.sample(self.coco.image_ids.tolist(), 1)[0]
        image_id = coco_id + self.coco.num_images
        combined_image = self.combined.load_image(image_id)
        mask, combind_class_ids = self.combined.load_mask(image_id)
        combind_class_ids = (np.array(combind_class_ids) - 1).tolist()
        _, coco_class_ids = self.coco.load_mask(coco_id)
        np.testing.assert_equal(combind_class_ids, coco_class_ids, 'ID mapping failed')
        if VISUALIZE:
            visualize_mask(combined_image, mask, coco_class_ids, COCO_CLASS_NAMES)

    def test_two_coco_datasets(self):
        self.combined.add_dataset('coco1', self.coco)
        self.combined.prepare()
        self.assertEqual(self.combined.num_images, 2 * len(self.coco.image_ids), 'different number of images')
        # image id from first dataset
        image_id = random.sample(self.coco.image_ids.tolist(), 1)[0]
        combined_image = self.combined.load_image(image_id)
        coco_image = self.coco.load_image(image_id)
        np.testing.assert_equal(combined_image, coco_image, 'different images returned')
        # now take the same image from the second dataset
        image_id = image_id + self.coco.num_images
        combined_image = self.combined.load_image(image_id)
        np.testing.assert_equal(combined_image, coco_image, 'different images returned')
        # TODO: test masks are equal
        image = self.combined.load_image(image_id)
        mask, class_ids = self.combined.load_mask(image_id)

    def test_coco_cityscape(self):
        self.cityscape = CityScapeDataset('/home/fattouhm/datasets/cityscape', 'val')
        self.combined.add_dataset('cityscape', self.cityscape)
        self.combined.prepare()

        coco_id = random.sample(self.coco.image_ids.tolist(), 1)[0]
        coco_image = self.coco.load_image(coco_id)
        combined_image = self.combined.load_image(coco_id)
        np.testing.assert_equal(combined_image, coco_image, 'different images returned')

        coco_mask, coco_ids = self.coco.load_mask(coco_id)
        combined_mask, combined_ids = self.combined.load_mask(coco_id)
        np.testing.assert_equal(combined_mask, coco_mask, 'different masks returned')
        np.testing.assert_equal(combined_ids, coco_ids, 'different ids returned')

        cityscape_id = random.sample(self.cityscape.image_ids.tolist(), 1)[0]
        cityscape_image = self.cityscape.load_image(cityscape_id)
        combined_image = self.combined.load_image(cityscape_id + self.coco.num_images)
        np.testing.assert_equal(combined_image, cityscape_image, 'different images returned')

        cityscape_mask, cityscape_ids = self.cityscape.load_mask(cityscape_id)
        combined_mask, combined_ids = self.combined.load_mask(cityscape_id + self.coco.num_images)
        np.testing.assert_equal(combined_mask, cityscape_mask, 'different masks returned')
        np.testing.assert_equal(combined_ids, cityscape_ids, 'different ids returned')

    def test_coco_cityscape_with_mapping(self):
        del self.coco, self.combined
        self.combined = CombinedDataset()
        with open('id_mapping.yaml', 'r') as fp:
            id_maps = yaml.load(fp)
            coco_map = id_maps['COCO']
            cityscape_map = id_maps['CITYSCAPE']
        self.coco = MyCocoDataset('/home/fattouhm/datasets/coco', class_map=coco_map)
        self.combined.add_dataset('coco', self.coco)
        self.cityscape = CityScapeDataset('/home/fattouhm/datasets/cityscape/', 'val', class_map=cityscape_map)
        self.combined.add_dataset('cityscape', self.cityscape)
        self.combined.prepare()

        coco_id = random.sample(self.coco.image_ids.tolist(), 1)[0]
        coco_image = self.coco.load_image(coco_id)
        combined_image = self.combined.load_image(coco_id)
        np.testing.assert_equal(combined_image, coco_image, 'different images returned')

        coco_mask, coco_ids = self.coco.load_mask(coco_id)
        combined_mask, combined_ids = self.combined.load_mask(coco_id)
        np.testing.assert_equal(combined_mask, coco_mask, 'different masks returned')
        np.testing.assert_equal(combined_ids, coco_ids, 'different ids returned')

        cityscape_id = random.sample(self.cityscape.image_ids.tolist(), 1)[0]
        cityscape_image = self.cityscape.load_image(cityscape_id)
        combined_image = self.combined.load_image(cityscape_id + self.coco.num_images)
        np.testing.assert_equal(combined_image, cityscape_image, 'different images returned')

        cityscape_mask, cityscape_ids = self.cityscape.load_mask(cityscape_id)
        combined_mask, combined_ids = self.combined.load_mask(cityscape_id + self.coco.num_images)
        np.testing.assert_equal(combined_mask, cityscape_mask, 'different masks returned')
        np.testing.assert_equal(combined_ids, cityscape_ids, 'different ids returned')
        if VISUALIZE:
            visualize_mask(combined_image, combined_mask, combined_ids, CITYSCAPE_TRAIN_NAMES)


if __name__ == '__main__':
    unittest.main()

