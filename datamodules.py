from torch.utils.data import Dataset
from torchvision import datasets
import torch
from PIL import Image
from torchvision import transforms
import os


class SamDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self):

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

        self.label = {
            "COCO_val2014_000000491213": [8],
            "COCO_val2014_000000516641": [4, 6],
            "COCO_val2014_000000533137": [1, 8, 10, 11],
            "COCO_val2014_000000542634": [3, 5, 6, 7, 8, 9, 11, 12, 22, 30, 37, 44, 62],
            "COCO_val2014_000000543112": [3, 8, 44, 72],
            "COCO_val2014_000000550576": [3],
            "COCO_val2014_000000581062": [0, 3, 11, 28],
            "COCO_val2014_000000576820": [0, 10]
        }

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

        return combined_tensor, label

    def __len__(self):
        return self.dataset_size


if __name__ == "__main__":
    dataset = SamDataset()
    print(len(dataset))

    for i, j in dataset:
        print(i.shape,  j)
        break
