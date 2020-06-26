import torch
import torchvision
import pandas as pd
import os
from PIL import Image
from torchvision.transforms import transforms

from hyper_parameters import PHASE
from hyper_parameters import ANNOTATION_FILE_NAME
from hyper_parameters import TRANSFORM_RESIZE_X
from hyper_parameters import TRANSFORM_RESIZE_Y
from hyper_parameters import BATCH_SIZE

TRAIN_ANNO_FILE = ANNOTATION_FILE_NAME % PHASE[0]
TEST_ANNO_FILE = ANNOTATION_FILE_NAME % PHASE[1]


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, anno_filepath, transform=None):

        super(MyDataset, self).__init__()
        self.anno_filepath = anno_filepath
        self.transform = transform

        if not os.path.isfile(self.anno_filepath):
            print("ERROR: anno file not found")

        self.anno_contents = pd.read_csv(self.anno_filepath, index_col=0)

        self.size = len(self.anno_contents)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image_path = self.anno_contents['image_path'][idx]

        if not os.path.isfile(image_path):
            print(f"ERROR: {image_path} not found")
            return None

        # image = cv2.imread(image_path, 0)
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        class_id = int(self.anno_contents.iloc[idx]['class_id'])

        sample = {'image': image, 'class_id': class_id}
        return sample


# transforms
train_transforms = transforms.Compose([
    transforms.Resize((TRANSFORM_RESIZE_Y, TRANSFORM_RESIZE_X)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
test_transforms = torchvision.transforms.transforms.Compose([
    transforms.Resize((TRANSFORM_RESIZE_Y, TRANSFORM_RESIZE_X)),
    transforms.ToTensor()
])

# datasets
train_dataset = MyDataset(anno_filepath=TRAIN_ANNO_FILE,
                          transform=train_transforms)
test_dataset = MyDataset(anno_filepath=TEST_ANNO_FILE,
                         transform=test_transforms)

# data_loaders
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset)

data_loaders = {PHASE[0]: train_loader, PHASE[1]: test_loader}
