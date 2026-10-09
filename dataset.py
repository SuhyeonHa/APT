import os
from torch.utils.data import Dataset
from PIL import Image
import random
import cv2
import numpy as np
import torch
from torchvision.transforms import transforms



def create_random_shapes_mask(height, width, num_shapes=5, max_iterations=10):
    mask = np.zeros((height, width), dtype=np.uint8)
    num_shapes = np.random.randint(2, num_shapes)
    for _ in range(num_shapes):
        seed_point = (np.random.randint(0, width), np.random.randint(0, height))
        
        mask[seed_point[1], seed_point[0]] = 1
        
        iterations = np.random.randint(1, max_iterations)
        
        kernel_size = np.random.randint(3, 15)
        kernel_type = random.choice([cv2.MORPH_ELLIPSE, cv2.MORPH_RECT, cv2.MORPH_ELLIPSE])
        kernel = cv2.getStructuringElement(kernel_type, (kernel_size, kernel_size))
        
        mask = cv2.dilate(mask, kernel, iterations=iterations)
    
    mask = np.clip(mask, 0, 1)

    mask_tensor = torch.tensor(mask, dtype=torch.float32).view(1, height, width)
    return mask_tensor


class CocoDataset(Dataset):
    def __init__(self, data_root, mask_path, mode="train", size=256) -> None:
        super().__init__()
        self.data_root = data_root
        self.size = size
        self.mode = mode
        self.mask_path = mask_path
        self.images_list = os.listdir(os.path.join(self.data_root, mode+'2017'))
        self.mask_list = os.listdir(mask_path)
        self.images_list.sort()
        self.transform = transforms.Compose([
            transforms.Resize((self.size, self.size)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((self.size, self.size)),
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        image_name = self.images_list[index]
        image = Image.open(os.path.join(self.data_root, self.mode+'2017', image_name)).convert("RGB")
        mask = random.sample(self.mask_list, k=1)[0]
        mask_image_list = [res for res in os.listdir(os.path.join(self.mask_path, mask)) if res.endswith('.png')]
        rand_num = random.random()
        if len(mask_image_list) > 0 and rand_num <= 0.9:
            mask_image_path = os.path.join(self.mask_path, mask, random.sample(mask_image_list, k=1)[0])
            random_mask = Image.open(mask_image_path).convert("L")
            random_mask = self.mask_transform(random_mask)
        else:
            random_mask = create_random_shapes_mask(height=self.size, width=self.size)
        image = self.transform(image)

        return {"image": image, 
                "random_mask": random_mask}

    def __len__(self):
        return len(self.images_list)


class ImageDataset(Dataset):
    def __init__(self, data_root, mode="train", size=256) -> None:
        super().__init__()
        self.data_root = data_root
        self.size = size
        self.mode = mode
        self.images_list = os.listdir(os.path.join(self.data_root, "original"))
        self.images_list.sort()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        image_name = self.images_list[index]
        image = Image.open(os.path.join(self.data_root,  "original", image_name)).convert("RGB").resize((self.size, self.size))
        mask = Image.open(os.path.join(self.data_root,  "mask", image_name)).convert("RGB").resize((self.size, self.size))
        generated_image = Image.open(os.path.join(self.data_root, "generated", image_name)).convert("RGB").resize((self.size, self.size))
        image = self.transform(image)
        generated_image = self.transform(generated_image)
        mask = self.mask_transform(mask)

        return {"image": image, 
                "generated_image": generated_image,
                "mask": mask,
                "image_name": image_name}

    def __len__(self):
        return len(self.images_list)

    
    

def image_collate_fn(data):       
    images = torch.stack([example["image"] for example in data])
    generated_images = torch.stack([example["generated_image"] for example in data])
    masks = torch.stack([example["mask"] for example in data])
    image_names = [example["image_name"] for example in data]

    return {
            "images": images,
            "generated_images": generated_images,
            "masks": masks,
            "image_names": image_names
    }


    

class AGEDataset(Dataset):
    def __init__(self, data_root, norm_type="imagenet", mode="val", size=512) -> None:
        super().__init__()
        self.data_root = data_root
        self.size = size
        self.mode = mode
        valid_exts = (".jpg", ".jpeg", ".png")
        file_list = os.listdir(self.data_root)
        self.images_list = [f for f in file_list if f.lower().endswith(valid_exts)]
        self.images_list.sort()
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        if norm_type == "imagenet":
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        elif norm_type == "rescale": # [0, 1] -> [-1, 1]
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
        ])

    def __getitem__(self, index):
        image_name = self.images_list[index]
        image = Image.open(os.path.join(self.data_root, image_name)).convert("RGB").resize((self.size, self.size))
        mask = Image.open(os.path.join(self.data_root + "-Mask", image_name)).convert("L").resize((self.size, self.size))
        image = self.transform(image)
        mask = self.mask_transform(mask)

        return {"image": image, 
                "mask": mask,
                "image_name": image_name}

    def __len__(self):
        return len(self.images_list)
    
def age_collate_fn(data):       
    images = torch.stack([example["image"] for example in data])
    masks = torch.stack([example["mask"] for example in data])
    image_names = [example["image_name"] for example in data]

    return {
            "images": images,
            "masks": masks,
            "image_names": image_names
    }