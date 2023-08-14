from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import torch
import json
import os


class SamDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, type):

        self.image_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

        with open("label.json", 'r') as j:
            self.label = json.load(j)

        self.label = self.label[type]

        self.dataset = []

        for image_id in self.label.keys():
            mask_path = os.path.join("coco2014/mask", image_id)

            for mask_image_name in os.listdir(mask_path):
                if mask_image_name.endswith(".png"):
                    self.dataset.append(
                        {"image_id": image_id, "mask_image_name": mask_image_name})

        self.dataset_size = len(self.dataset)

    def __getitem__(self, i):

        dataset = self.dataset[i]
        image_id = dataset["image_id"]
        mask_image_name = dataset["mask_image_name"]

        label = int(os.path.splitext(mask_image_name)
                    [0]) in self.label[image_id]

        mask_path = os.path.join("coco2014/mask", image_id, mask_image_name)
        out_painted = os.path.join("coco2014/out_painted", image_id + ".jpg")

        rgb_image = Image.open(out_painted)
        mask_image = Image.open(mask_path).convert("L")

        rgb_tensor = self.image_transform(rgb_image)
        mask_tensor = self.mask_transform(mask_image)

        combined_tensor = torch.cat([rgb_tensor, mask_tensor], dim=0)

        return combined_tensor, int(label)

    def __len__(self):
        return self.dataset_size


if __name__ == "__main__":
    dataset = SamDataset()
    print(len(dataset))

    for i, j in dataset:
        print(i.shape, j)
        break
