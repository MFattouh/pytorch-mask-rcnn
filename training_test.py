import yaml
from datasets import CombinedDataset, MyCocoDataset, CityScapeDataset
from config import Config
import model as modellib
from torch import manual_seed as cpu_seed
from torch.cuda import manual_seed_all as gpu_seed
from numpy.random import seed as np_seed

RANDOM_SEED = 20180705

np_seed(RANDOM_SEED)
cpu_seed(RANDOM_SEED)
gpu_seed(RANDOM_SEED)


def create_coco_cityscape_with_mapping(subset='val'):
    assert subset in ['train', 'val']
    combined = CombinedDataset()
    with open('id_mapping.yaml', 'r') as fp:
        id_maps = yaml.load(fp)
        coco_map = id_maps['COCO']
        cityscape_map = id_maps['CITYSCAPE']

    cityscape = CityScapeDataset('/home/fattouhm/datasets/cityscape/', subset, class_map=cityscape_map)
    combined.add_dataset('cityscape', cityscape)
    if subset == 'val':
        subset == 'minival'
    coco = MyCocoDataset('/home/fattouhm/datasets/coco', subset=subset, class_map=coco_map)
    combined.add_dataset('coco', coco)
    combined.prepare()

    return combined


def train():
    class InferenceConfig(Config):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 2
        # We use one GPU with 8GB memory, which can fit one image.
        # Adjust down if you use a smaller GPU.
        # IMAGES_PER_GPU = 16
        IMAGES_PER_GPU = 32
        DETECTION_MIN_CONFIDENCE = 0
        NAME = "test"
        # Start from coco weights
        MODEL_PATH = 'mask_rcnn_coco.pth'

        # Number of classes (including background)
        NUM_CLASSES = 1 + 19  #  Cityscape TrainIDs are 19
    config = InferenceConfig()
    config.display()

    # Create model
    model = modellib.MaskRCNN(config=config, model_dir='/home/fattouhm/mask-rcnn/test')
    if config.GPU_COUNT:
        model.cuda()

    # load model weights
    print("Loading weights ", config.MODEL_PATH)
    model.load_weights(config.MODEL_PATH, layers='4+')

    train_dataset = create_coco_cityscape_with_mapping('train')
    val_dataset = create_coco_cityscape_with_mapping('val')
    print("Fine tune Resnet stage 4 and up")
    model.train_model(train_dataset, val_dataset,
                      learning_rate=config.LEARNING_RATE,
                      epochs=120,
                      layers='4+')


if __name__ == '__main__':
    train()
