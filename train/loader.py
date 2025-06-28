import os
import json
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class CrowdHumanDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        with open(annotation_file, 'r') as f:
            self.annotations = [json.loads(line) for line in f]

        self.samples = []
        for ann in self.annotations:
            image_name = ann['ID'] + ".jpg"
            bboxes = [h['bbox'] for h in ann['gtboxes'] if h['tag'] == 'person']
            self.samples.append((image_name, bboxes))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, bboxes = self.samples[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        label = len(bboxes)  # number of people
        if self.transform:
            image = self.transform(image)
        return image, label
