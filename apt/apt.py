import os
import torch
from diffusers import DiffusionPipeline, EulerDiscreteScheduler
import timm
import torch.nn.functional as F
import random
import torch.optim as optim
import numpy as np
import lpips
from .helper import norm_imagenet, denorm_imagenet
from .train_decoder import ShallowUpDecoder


epsilon = 1e-6

class APT:
    def __init__(self, args):
        self.args = args
        self.weights_dir = getattr(self.args, 'weights_dir', 'weights')

        # Initialize networks
        scheduler = EulerDiscreteScheduler.from_pretrained(
            self.args.vae_model_name, subfolder="scheduler",
        )
        self.pipe = DiffusionPipeline.from_pretrained(
            self.args.vae_model_name,
            scheduler=scheduler,
        )
        self.pipe.vae = self.pipe.vae.to(self.args.device)
        self.image_encoder = timm.create_model(
            'convnext_small.dinov3_lvd1689m',
            pretrained=True,
            features_only=True
        ).to(self.args.device)
        self.mask_refiner = ShallowUpDecoder().to(self.args.device)
        self.mask_refiner.load_state_dict(torch.load(os.path.join(self.weights_dir, 'shallow_refiner_6.pth'), map_location=self.args.device))

        for param in self.image_encoder.parameters():
            param.requires_grad = False

        self.pipe.vae.requires_grad_(False)
        self.pipe.unet.requires_grad_(False)
        self.pipe.text_encoder.requires_grad_(False)
        self.pipe.vae.eval()
        self.pipe.unet.eval()
        self.pipe.text_encoder.eval()
        del self.pipe.unet
        del self.pipe.text_encoder
        torch.cuda.empty_cache()
        
        # load the pre-generated anchor vectors
        anchor_path = os.path.join(self.weights_dir, f'anchor_vecs_{self.args.feature_dim}.pt')
        print(f"Loading anchor vectors from: {anchor_path}")
        self.direction_vectors = torch.load(anchor_path).to(self.args.device)
        self.num_patches = (self.args.image_size // 14) ** 2

        self.loss_fn_vgg = lpips.LPIPS(net='alex').to(self.args.device)
        self.loss_fn_vgg.eval()

    def _create_random_mask(self, img_pt, num_masks=1, mask_percentage=0.1, max_attempts=100):
        _, _, height, width = img_pt.shape
        mask_area = int(height * width * mask_percentage)
        masks = torch.zeros((num_masks, 1, height, width), dtype=img_pt.dtype)

        if mask_percentage >= 0.999:
            # Full mask for entire image
            return torch.ones((num_masks, 1, height, width), dtype=img_pt.dtype).to(img_pt.device)

        for ii in range(num_masks):
            placed = False
            attempts = 0
            while not placed and attempts < max_attempts:
                attempts += 1

                max_dim = int(mask_area ** 0.5)
                mask_width = random.randint(1, max_dim)
                mask_height = mask_area // mask_width

                aspect_ratio = mask_width / mask_height if mask_height != 0 else 0
                if 0.25 <= aspect_ratio <= 4:  # Looser ratio constraint
                    if mask_height <= height and mask_width <= width:
                        x_start = random.randint(0, width - mask_width)
                        y_start = random.randint(0, height - mask_height)
                        overlap = False
                        for jj in range(ii):
                            if torch.sum(masks[jj, :, y_start:y_start + mask_height, x_start:x_start + mask_width]) > 0:
                                overlap = True
                                break
                        if not overlap:
                            masks[ii, :, y_start:y_start + mask_height, x_start:x_start + mask_width] = 1
                            placed = True

            if not placed:
                print(f"Warning: Failed to place mask {ii}, using fallback.")
                center_h = height // 2
                center_w = width // 2
                half_area = int((mask_area // 2) ** 0.5)
                h_half = min(center_h, half_area)
                w_half = min(center_w, half_area)
                masks[ii, :, center_h - h_half:center_h + h_half, center_w - w_half:center_w + w_half] = 1

        return masks.to(img_pt.device)

    def embed_watermark(self, original: torch.Tensor, img_size: int) -> torch.Tensor:
        original = original.to(self.args.device)
        image = F.interpolate(original, size=(self.args.vae_image_size, self.args.vae_image_size), mode="bilinear", align_corners=False)
        latent = self.pipe.vae.encode(2*image-1).latent_dist.sample()
        
        delta_m = torch.zeros_like(latent, requires_grad=True)
        optimizer = optim.Adam([delta_m], lr=self.args.lr)

        # Training loop
        for step in range(self.args.steps):
            optimizer.zero_grad()

            original_mask = self._create_random_mask(image, num_masks=1, mask_percentage=self.args.mask_percentage)
            original_mask = original_mask.to(self.args.device)
            target_mask = original_mask * 2 - 1 # Convert to {-1, 1}

            if random.random() < 0.5:
                original_mask = 1 - original_mask

            image_512 = F.interpolate(original, size=(self.args.vae_image_size, self.args.vae_image_size), mode="bilinear", align_corners=False)

            perturbed_latent = latent + delta_m

            watermarked_image = self.pipe.vae.decode(perturbed_latent).sample
            watermarked_image = (watermarked_image + 1) / 2

            # Patch noise injection
            latent_mask = F.interpolate(original_mask, size=(64, 64), mode="bilinear", align_corners=False)

            std_val_0 = random.uniform(self.args.eps0_std[0], self.args.eps0_std[1])
            eps0 = torch.randn_like(perturbed_latent) * std_val_0
            perturbed_latent_1 = (perturbed_latent + eps0) * latent_mask + perturbed_latent * (1 - latent_mask)
            watermarked_image_1 = self.pipe.vae.decode(perturbed_latent_1).sample
            watermarked_image_1 = (watermarked_image_1 + 1) / 2
            watermarked_image_1 = torch.clamp(watermarked_image_1, 0.0, 1.0)

            # Compute losses
            image = F.interpolate(original, size=(img_size, img_size), mode="bilinear", align_corners=False)
            target_mask = F.interpolate(target_mask, size=(img_size, img_size), mode="bilinear", align_corners=False)

            watermarked_image = F.interpolate(watermarked_image, size=(img_size, img_size), mode="bilinear", align_corners=False)

            image = norm_imagenet(image)
            watermarked_image = norm_imagenet(watermarked_image)

            features = self.image_encoder(image)[self.args.feat_layer]

            B, C, H, W = features.shape # [1, 192, 32, 32]
            features = features.permute(0, 2, 3, 1).view(B, H * W, C)
            features_norm = features / (torch.norm(features, p=2, dim=-1, keepdim=True) + epsilon)
            base_cos_sim = torch.matmul(features_norm, self.direction_vectors.T)

            if step == 0 or (step+1) % 100 == 0:
                print("Base Cosine Similarity - Min: {}, Max: {}, Mean: {}".format(
                    torch.min(base_cos_sim), torch.max(base_cos_sim), torch.mean(base_cos_sim)))
                
            noise_floor = torch.max(base_cos_sim)

            features = self.image_encoder(watermarked_image)[self.args.feat_layer]

            B, C, H, W = features.shape
            features = features.permute(0, 2, 3, 1).view(B, H * W, C)
            features_norm = features / (torch.norm(features, p=2, dim=-1, keepdim=True) + epsilon)
            cos_sim = torch.matmul(features_norm, self.direction_vectors.T)
            if step == 0 or (step+1) % 100 == 0:
                print("Cosine Similarity - Min: {}, Max: {}, Mean: {}".format(
                    torch.min(cos_sim), torch.max(cos_sim), torch.mean(cos_sim)))

            watermarked_image_1 = F.interpolate(watermarked_image_1, size=(img_size, img_size), mode="bilinear", align_corners=False)
            watermarked_image_1 = norm_imagenet(watermarked_image_1)

            features = self.image_encoder(watermarked_image_1)[self.args.feat_layer]
            features = features.permute(0, 2, 3, 1).view(B, H * W, C)
            features_norm = features / (torch.norm(features, p=2, dim=-1, keepdim=True) + epsilon)
            cos_sim_1 = torch.matmul(features_norm, self.direction_vectors.T)
            if step == 0 or (step+1) % 100 == 0:
                print("Cosine Similarity (Noisy) - Min: {}, Max: {}, Mean: {}".format(
                    torch.min(cos_sim_1), torch.max(cos_sim_1), torch.mean(cos_sim_1)))

            B = cos_sim.shape[0]
            H = W = int(cos_sim.shape[1] ** 0.5)

            target_cosine = self.args.target_cossim 

            if step==0:
                print(f"Noise Floor Cosine Similarity: {noise_floor:.4f}")
                print(f"Target Cosine Similarity: {target_cosine:.4f}")
            
            loss_m = torch.mean(F.relu(target_cosine - cos_sim))
            loss_m1 = torch.mean(F.relu(target_cosine - cos_sim_1))
            loss_h = self._hard_negative_mining_loss(cos_sim, target_cosine)
            loss_h1 = self._hard_negative_mining_loss(cos_sim_1, target_cosine)

            watermarked_image_1 = denorm_imagenet(watermarked_image_1)
            image = denorm_imagenet(image)
            watermarked_image = denorm_imagenet(watermarked_image)

            loss_psnr = self._psnr_loss(watermarked_image, image)
            loss_lpips = self._lpips_loss(watermarked_image, image)

            total_loss = self.args.lambda_clean * loss_m + \
                         self.args.lambda_noisy * loss_m1 + \
                         loss_h + loss_h1 + \
                         self.args.lambda_p * loss_psnr + \
                         self.args.lambda_i * loss_lpips

            total_loss.backward()
            optimizer.step()

            if step == 0 or (step+1) % 50 == 0:
                psnr_val = self._compute_psnr(watermarked_image.detach(), image.detach())
                print(f"Step {step+1}, Loss: {total_loss.item():.4f}, PSNR: {psnr_val:.2f}")
                print(f"Mask loss: {loss_m.item():.4f}, Mask1 loss: {loss_m1.item():.4f}")
                print(f"Hard Neg Loss: {loss_h.item():.4f}, Hard Neg1 Loss: {loss_h1.item():.4f}")
                print(f"PSNR Loss: {loss_psnr.item():.4f}, LPIPS Loss: {loss_lpips.item():.4f}")

        with torch.no_grad():
            latent_wm = latent + delta_m
            rec_wm = self.pipe.vae.decode(latent_wm).sample
            rec_wm = (rec_wm + 1) / 2

            final_delta = torch.clamp(rec_wm - image_512, -self.args.epsilon, self.args.epsilon)
            final_images = torch.clamp(image_512 + final_delta, 0, 1)
        return final_images.detach(), final_delta.detach() 
        
    def decode_watermark(self, watermarked_image: torch.Tensor, use_refiner: bool = True) -> torch.Tensor:
        watermarked_image = watermarked_image.to(self.args.device)
        smoother = torch.nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.mask_refiner.eval()

        with torch.no_grad():
            watermarked_image = norm_imagenet(watermarked_image)
            features = self.image_encoder(watermarked_image)[self.args.feat_layer]
            features = smoother(features)

            B, C, H, W = features.shape
            features = features.permute(0, 2, 3, 1).view(B, H * W, C)

            epsilon = 1e-6
            features_norm = features / (torch.norm(features, p=2, dim=-1, keepdim=True) + epsilon)
            dot_products = torch.matmul(features_norm, self.direction_vectors.T)

            B = dot_products.shape[0]
            H = W = int(dot_products.shape[1] ** 0.5)
            grid = dot_products.view(B, H, W).unsqueeze(0) # [1, 1024, 1] -> [1, 1, 32, 32]

            if use_refiner:
                logits = self.mask_refiner(grid)
            else:
                logits = F.interpolate(grid, size=self.args.image_size, mode='bilinear', align_corners=False)
                logits = logits * self.args.temperature

            confidence_map = torch.sigmoid(logits)
            binary_prediction = (confidence_map > 0.5).float()

        return grid, confidence_map, binary_prediction
    
    def _psnr_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Negative PSNR loss (Equation 5)"""
        mse = F.mse_loss(pred, target)
        psnr = -10 * torch.log10(mse + 1e-8)
        return -psnr
    
    def _lpips_loss(self, pred, target):
        pred_norm = pred * 2 - 1 # [-1, 1]
        target_norm = target * 2 - 1 # [-1, 1]
        return self.loss_fn_vgg(pred_norm, target_norm).mean()
    
    def _compute_psnr(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Compute PSNR between images"""
        mse = F.mse_loss(pred, target).item()
        if mse == 0:
            return 100.0
        return 20 * np.log10(1.0 / np.sqrt(mse))
    
    def _dice_loss(self, cos_sim, gt_mask, smooth=1e-5):
        B = cos_sim.shape[0]
        H = W = int(cos_sim.shape[1] ** 0.5)
        grid = cos_sim.view(B, H, W).unsqueeze(0)
        grid = F.interpolate(grid, size=self.args.image_size, mode='bilinear', align_corners=False)
        pred = torch.sigmoid(grid * self.args.temperature)
       
        pred = pred.view(-1)
        target = gt_mask.view(-1)

        intersection = (pred * target).sum()
        dice_coeff = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        
        return 1 - dice_coeff
    
    def _hard_negative_mining_loss(self, scores, target_val, k_percent=0.15):
        hinge = F.relu(target_val - scores).view(-1)
        num_hard = max(1, int(hinge.numel() * k_percent))
        top_k_loss, _ = torch.topk(hinge, num_hard)
        return top_k_loss.mean()
