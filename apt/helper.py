
import os
import torch
from PIL import Image
import torchvision.transforms as transforms
import torchvision

# Load images 
class Denormalize(transforms.Normalize):
    def __init__(self, mean, std):
        mean = torch.tensor(mean)
        std = torch.tensor(std)
        self.mean_rev = -mean / std
        self.std_rev = 1 / std
        super().__init__(mean=self.mean_rev, std=self.std_rev)

norm_imagenet = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
denorm_imagenet = Denormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

def load_images_from_path(path, num_images=None, transform=None) -> torch.Tensor:
    images = []
    file_names = []
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    
    for filename in sorted(os.listdir(path)):

        if num_images is not None and len(images) >= num_images:
            break

        if filename.lower().endswith(valid_extensions):
            image_path = os.path.join(path, filename)
            try:
                image = Image.open(image_path).convert("RGB")
                images.append(transform(image))
                file_names.append(filename)
            except Exception as e:
                print(f"Warning: Could not load image {image_path}. Error: {e}")

    if not images:
        raise FileNotFoundError(f"No valid images found in the specified path: {path}")

    return torch.stack(images), file_names


def load_image(path: str, transform=None) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    return transform(image).unsqueeze(0)

def load_mask(path: str, transform=None) -> torch.Tensor:
    mask = Image.open(path).convert("L")
    return transform(mask).unsqueeze(0)

class Tee:
    def __init__(self, stream1, stream2):
        self.stream1 = stream1
        self.stream2 = stream2

    def write(self, message):
        self.stream1.write(message)
        self.stream2.write(message)
        self.flush()

    def flush(self):
        self.stream1.flush()
        self.stream2.flush()

# Save images
def save_images(results, filename, save_dir):
    names = ["original", "watermarked", "prediction", "watermark"]

    for img, name in zip(results, names):
        output_dir = os.path.join(save_dir, name)
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"{filename}")
        if name == "watermark":
            img = img*5
        torchvision.utils.save_image(img, save_path)

    # Save all images at once
    combined = []
    for tensor in results:
        if tensor.shape[1] == 1:
            tensor = tensor.repeat(1, 3, 1, 1)
        combined.append(tensor)

    combined = torch.cat(combined, dim=0)
    os.makedirs(os.path.join(save_dir, "results"), exist_ok=True)
    save_path = os.path.join(save_dir, "results", f"{filename}")
    torchvision.utils.save_image(combined, save_path, nrow=3)