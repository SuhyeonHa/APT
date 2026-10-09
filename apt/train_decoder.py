import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from tqdm import tqdm
import os
from torchvision.datasets import CocoDetection
from pycocotools import mask as maskUtils
import numpy as np
import random
from torchvision.utils import save_image
import timm
import torch.nn.functional as F
import albumentations as A

def snap_mask_to_edge(mask, margin=3):
    mask[..., :margin, :] = torch.max(mask[..., :margin, :], mask[..., margin:margin+1, :])
    mask[..., -margin:, :] = torch.max(mask[..., -margin:, :], mask[..., -margin-1:-margin, :])
    mask[..., :, :margin] = torch.max(mask[..., :, :margin], mask[..., :, margin:margin+1])
    mask[..., :, -margin:] = torch.max(mask[..., :, -margin:], mask[..., :, -margin-1:-margin])
    
    return mask

class CocoImageIDWrapper(CocoDetection):
    def __init__(self, root, annFile, transform=None, mask_transform=None, random_nb_object=True, max_nb_masks=4, multi_w=False, min_mask_ratio=0.02):
        super().__init__(root, annFile, transform=transform, target_transform=mask_transform)
        self.random_nb_object = random_nb_object
        self.max_nb_masks = max_nb_masks
        self.multi_w = multi_w
        self.min_mask_ratio = min_mask_ratio

    def _generate_rectangular_mask(self, h, w):
        rh = random.randint(int(h * 0.1), int(h * 0.5))
        rw = random.randint(int(w * 0.1), int(w * 0.5))

        if random.random() < 0.25:
            side = random.randint(0, 3)
            if side == 0:   y1, x1 = 0,      random.randint(0, w - rw)  # top
            elif side == 1: y1, x1 = h - rh, random.randint(0, w - rw)  # bottom
            elif side == 2: y1, x1 = random.randint(0, h - rh), 0       # left
            else:           y1, x1 = random.randint(0, h - rh), w - rw  # right
        else:
            y1 = random.randint(0, h - rh)
            x1 = random.randint(0, w - rw)

        mask = np.zeros((h, w), dtype=np.float32)
        mask[y1:y1+rh, x1:x1+rw] = 1.0
        return torch.tensor(mask, dtype=torch.float32)[None, ...]  # (1, H, W)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, np.ndarray]:
        if not isinstance(index, int):
            raise ValueError(f"Index must be of type integer, got {type(index)} instead.")

        id = self.ids[index]
        img = self._load_image(id)
        img_info = self.coco.loadImgs(id)[0]
        h, w = img_info['height'], img_info['width']

        mask = None
        if random.random() < 0.5:
            mask = self._generate_rectangular_mask(h, w)
        else:
            mask = None
            while mask is None:
                random_index = random.randint(0, len(self.ids) - 1)
                mask = self._load_mask(self.ids[random_index])

        if mask is not None:
            mask = snap_mask_to_edge(mask, margin=5)

        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        return img, mask

    def _load_mask(self, id):
        anns = self.coco.loadAnns(self.coco.getAnnIds(id))
        if not anns:
            return None  

        img_info = self.coco.loadImgs(id)[0]
        original_height = img_info['height']
        original_width = img_info['width']

        masks = []
        if not self.multi_w:
            random.shuffle(anns)
            min_area = original_height * original_width * self.min_mask_ratio
            for ann in anns:
                rle = self.coco.annToRLE(ann)
                m = maskUtils.decode(rle).astype(np.float32)
                if m.sum() < min_area:
                    continue
                return torch.tensor(m, dtype=torch.float32)[None, ...]
            return None
        else:
            anns = anns[:self.max_nb_masks]
            for ann in anns:
                rle = self.coco.annToRLE(ann)
                m = maskUtils.decode(rle)
                masks.append(m)
            if masks:
                masks = np.stack(masks, axis=0)
                masks = torch.tensor(masks, dtype=torch.bool)
                if masks.shape[0] < self.max_nb_masks:
                    additional_masks_count = self.max_nb_masks - masks.shape[0]
                    additional_masks = torch.zeros((additional_masks_count, original_height, original_width), dtype=torch.bool)
                    masks = torch.cat([masks, additional_masks], dim=0)
            else:
                masks = torch.zeros((self.max_nb_masks, original_height, original_width), dtype=torch.bool)
            return masks

def custom_collate(batch: list) -> tuple[torch.Tensor, torch.Tensor]:
    batch = [item for item in batch if item is not None]
    if not batch:
        return torch.tensor([]), torch.tensor([])
    
    images, masks = zip(*batch)
    images = torch.stack(images)
    
    max_masks = max(mask.shape[0] for mask in masks)
    if max_masks == 1:
        masks = torch.stack(masks)
        return images, masks

class ShallowUpDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Stage 1: 32 -> 64
        self.up1   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1a = nn.Conv2d(1,  32, kernel_size=3, padding=1)
        self.norm1a = nn.LayerNorm([32, 64, 64])
        self.conv1b = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.norm1b = nn.LayerNorm([32, 64, 64])

        # Stage 2: 64 -> 256
        self.up2   = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.conv2a = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.norm2a = nn.LayerNorm([16, 256, 256])
        self.conv2b = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.norm2b = nn.LayerNorm([16, 256, 256])

        self.out_conv = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        h = self.up1(x)
        h = F.relu(self.norm1a(self.conv1a(h)), inplace=True)
        h = F.relu(self.norm1b(self.conv1b(h)), inplace=True)

        h = self.up2(h)
        h = F.relu(self.norm2a(self.conv2a(h)), inplace=True)
        h = F.relu(self.norm2b(self.conv2b(h)), inplace=True)

        return self.out_conv(h)
    
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        num1 = 2. * (probs * targets).sum(dim=(1, 2, 3)) + self.smooth
        den1 = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + self.smooth
        dice1 = (num1 / den1).mean()

        inv_probs, inv_targets = 1. - probs, 1. - targets
        num0 = 2. * (inv_probs * inv_targets).sum(dim=(1, 2, 3)) + self.smooth
        den0 = inv_probs.sum(dim=(1, 2, 3)) + inv_targets.sum(dim=(1, 2, 3)) + self.smooth
        dice0 = (num0 / den0).mean()
        return 1. - (dice1 + dice0) / 2.

def generate_augmented_cossim(clean_cossim, masks, device, target_cossim=0.1):
    """
    Simulates watermarked cossim distribution for both spliced and spliceless cases.

    Empirical values (target_cossim=0.1):
      Spliced:    non-tampered mean=0.074 (0.74tc), gap=0.105 (1.05tc)
      Spliceless: non-tampered mean=0.043 (0.43tc), gap=0.076 (0.76tc)

    masks: 1=non-tampered (background), 0=tampered (object region)
    """
    B, _, H, W = clean_cossim.shape

    masks_resized = TF.resize(masks, [H, W], interpolation=TF.InterpolationMode.NEAREST)
    blurred_masks = TF.gaussian_blur(masks_resized, kernel_size=[5, 5], sigma=[0.5, 0.5])

    wm_offsets = torch.empty(B, 1, 1, 1, device=device).uniform_(target_cossim * 0.3, target_cossim * 0.9)
    high_baseline = clean_cossim + wm_offsets

    drop_margins = torch.empty(B, 1, 1, 1, device=device).uniform_(target_cossim * 0.7, target_cossim * 1.1)
    synthetic_tensor = high_baseline - drop_margins * (1.0 - blurred_masks)

    noise = torch.randn((B, 1, H, W), device=device) * 0.03
    synthetic_tensor = synthetic_tensor + noise

    return torch.clamp(synthetic_tensor, -1.0, 1.0)


def build_train_aug() -> A.Compose:
    """Randomly applies one augmentation per batch with p=0.5."""
    return A.Compose([
        A.OneOf([
            A.ImageCompression(quality_range=(40, 80), p=1.0),
            A.GaussianBlur(blur_limit=(3, 17), p=1.0),
            A.MedianBlur(blur_limit=(3, 7), p=1.0),
            A.RandomBrightnessContrast(brightness_limit=(-0.5, 1.0), contrast_limit=0, p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=(-0.5, 1.0), p=1.0),
            A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=(-50, 100), val_shift_limit=0, p=1.0),
            A.HueSaturationValue(hue_shift_limit=(-36, 36), sat_shift_limit=0, val_shift_limit=0, p=1.0),
        ], p=1.0),
    ], p=0.5)


def apply_aug_to_batch(images: torch.Tensor, transform: A.Compose) -> torch.Tensor:
    """
    images: [B, C, H, W] float tensor in [0, 1], on GPU
    Returns: augmented tensor of same shape on same device
    """
    device = images.device
    imgs_np = (images.cpu().permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
    augmented = [transform(image=img)["image"] for img in imgs_np]
    return torch.from_numpy(
        np.stack(augmented, axis=0).astype(np.float32) / 255.0
    ).permute(0, 3, 1, 2).to(device)


def train_framework():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    batch_size = 64
    epochs = 20
    lr = 3e-4
    img_size = 256
    feature_dim = 192

    save_dir = "./runs/mask_refiner"
    os.makedirs(save_dir, exist_ok=True)

    train_dir = "./data/coco-2017/train2017"
    ann_file = "./data/coco-2017/annotations/instances_train2017.json"
    
    image_transform = T.Compose([
        T.Resize((img_size, img_size)), 
        T.ToTensor()
    ])
    mask_transform = T.Compose([
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.NEAREST)
    ])
    
    dataset = CocoImageIDWrapper(
        root=train_dir, annFile=ann_file,
        transform=image_transform, mask_transform=mask_transform,
        random_nb_object=False, multi_w=False,
        min_mask_ratio=0.15
    )
    
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=8, pin_memory=True, drop_last=True,
        collate_fn=custom_collate 
    )
    
    model = ShallowUpDecoder().to(device)
    pos_weight = torch.tensor([0.3]).to(device)
    criterion_bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion_dice = DiceLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    aug_transform = build_train_aug()
    
    norm_imagenet = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    direction_vectors_path = f'./weights/anchor_vecs_{feature_dim}.pt'
    
    if os.path.exists(direction_vectors_path):
        direction_vectors = torch.load(direction_vectors_path).to(device)
    else:
        print("Warning: direction_vectors.pt not found. Using dummy vectors.")
        direction_vectors = torch.randn(256, feature_dim).to(device)

    image_encoder = timm.create_model(
        'convnext_small.dinov3_lvd1689m',
        pretrained=True,
        features_only=True
    ).to(device)
    image_encoder.eval()


    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, (images, masks) in enumerate(progress_bar):
            if masks is None or masks.numel() == 0:
                continue
            
            images = images.to(device)
            masks = (1-masks).to(device)

            images = apply_aug_to_batch(images, aug_transform)

            with torch.no_grad():
                norm_images = norm_imagenet(images)
                
                features = image_encoder(norm_images)[1]

                B, C, H, W = features.shape
                features_flat = features.permute(0, 2, 3, 1).reshape(B, H * W, C)
                
                epsilon = 1e-6
                features_norm = features_flat / (torch.norm(features_flat, p=2, dim=-1, keepdim=True) + epsilon)
                dot_products = torch.matmul(features_norm, direction_vectors.T)
                
                clean_cossim = dot_products[..., 0].view(B, 1, H, W)

            inputs = generate_augmented_cossim(clean_cossim, masks, device, target_cossim=0.1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion_bce(outputs, masks) + criterion_dice(outputs, masks)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

            if batch_idx == 0:
                num_samples = min(4, images.size(0))
                
                inputs_resized = TF.resize(inputs[:num_samples], [img_size, img_size], interpolation=TF.InterpolationMode.NEAREST)
                inputs_viz = ((inputs_resized + 1.0) / 2.0).repeat(1, 3, 1, 1)
                
                outputs_viz = torch.sigmoid(outputs[:num_samples]).repeat(1, 3, 1, 1)
                gt_viz = masks[:num_samples].repeat(1, 3, 1, 1)
                
                grid = torch.cat([images[:num_samples], inputs_viz, outputs_viz, gt_viz], dim=0)
                
                save_path = os.path.join(save_dir, f"epoch_{epoch+1:03d}.png")
                save_image(grid, save_path, nrow=num_samples)
            
        scheduler.step()
        print(f"Epoch [{epoch+1}/{epochs}] Average Loss: {epoch_loss/len(dataloader):.4f}, LR: {scheduler.get_last_lr()[0]:.2e}")
        torch.save(model.state_dict(), os.path.join(save_dir, f"shallow_refiner_{epoch+1}.pth"))

if __name__ == "__main__":
    model = ShallowUpDecoder()
    total = sum(p.numel() for p in model.parameters())
    if total >= 1_000_000:
        print(f"Total params: {total / 1_000_000:.2f}M ({total:,})")
    elif total >= 1_000:
        print(f"Total params: {total / 1_000:.2f}K ({total:,})")
    else:
        print(f"Total params: {total:,}")
    # train_framework()