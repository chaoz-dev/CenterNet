import json
import numpy as np
import pycocotools.coco as coco
import torch.utils.data as data

CUSTOM_DATASET_PATH = '/var/datasets/custom'
METADATA_FILE = 'metadata.json'


def custom_dataset():
    with open(CUSTOM_DATASET_PATH + '/' + METADATA_FILE) as metadata_file:
        metadata = json.load(metadata_file)

    return [entry['file_name'] for entry in metadata['images']]

class Custom(data.Dataset):
    default_resolution = [3840, 2160]
    num_classes = 80

    mean = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32).reshape(1, 1, 3)
    std  = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32).reshape(1, 1, 3)

    coco = coco.COCO()

    def __init__(self, opt, split):
        super(Custom, self).__init__()

        self.images = {i: path for i, path in enumerate(custom_dataset())}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int):
        return self.images[index]
