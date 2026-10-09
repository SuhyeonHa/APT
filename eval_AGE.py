import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

from PIL import Image
from torchvision.transforms import transforms as T
from tqdm import tqdm
import numpy as np
from diffusers import AutoencoderKL, StableDiffusionInpaintPipeline
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision.transforms import transforms, ToTensor
from piq import ssim, psnr, LPIPS

from evaluation import PixelF1, PixelAUC, PixelIOU, PixelAccuracy
from dataset import age_collate_fn, AGEDataset
import random
import yaml
import torch.nn.functional as F

from apt.apt import APT
from apt.main import Params
from omegaconf import OmegaConf

image_mean = torch.tensor([0.485, 0.456, 0.406])
image_std = torch.tensor([0.229, 0.224, 0.225])
denorm_imagenet = transforms.Normalize(-image_mean / image_std, 1 / image_std)

def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def convert_mask_to_rect(masks, scale_factor=1.0):
    B, C, H, W = masks.shape
    rect_masks = torch.zeros_like(masks)

    for i in range(B):
        mask_indices = torch.nonzero(masks[i, 0] > 0.5)

        if mask_indices.numel() == 0:
            continue

        y_min = mask_indices[:, 0].min().item()
        y_max = mask_indices[:, 0].max().item()
        x_min = mask_indices[:, 1].min().item()
        x_max = mask_indices[:, 1].max().item()
        h = y_max - y_min
        w = x_max - x_min
        cy = y_min + h / 2
        cx = x_min + w / 2

        new_h = h * scale_factor
        new_w = w * scale_factor

        y1 = int(max(0, cy - new_h / 2))
        y2 = int(min(H, cy + new_h / 2))
        x1 = int(max(0, cx - new_w / 2))
        x2 = int(min(W, cx + new_w / 2))

        rect_masks[i, 0, y1:y2, x1:x2] = 1.0

    return rect_masks

class Evaluation(object):
    def __init__(self, pred_path, gt_path, eval_size, end_idx=None) -> None:
        self.pred_path = pred_path
        self.gt_path = gt_path
        self.eval_size = eval_size
        self.end_idx = end_idx
        # instantiate metric calculators
        self.f1 = PixelF1()
        self.auc = PixelAUC()
        self.iou = PixelIOU()
        self.acc = PixelAccuracy()

    def run(self, save_path):
        total_f1 = []
        total_auc = []
        total_iou = []
        total_acc = []
        total_fpr = []
        total_pos_acc = []
        total_neg_acc = []
        total_b_acc = []

        pred_images_path = sorted(os.listdir(self.pred_path))
        if self.end_idx is not None:
            pred_images_path = pred_images_path[:self.end_idx]

        for pred_image_path in tqdm(pred_images_path):
            try:
                pred_image = Image.open(os.path.join(self.pred_path, pred_image_path)).convert("L")

                gt_image = Image.open(os.path.join(self.gt_path, pred_image_path)).convert("L")

                pred_image = pred_image.resize((self.eval_size, self.eval_size))
                gt_image = gt_image.resize((self.eval_size, self.eval_size))
            except Exception:
                continue

            pred_tensor = T.ToTensor()(pred_image).unsqueeze(0)
            gt_tensor = T.ToTensor()(gt_image).unsqueeze(0)

            gt_tensor = torch.nn.functional.interpolate(gt_tensor, (pred_tensor.size(2), pred_tensor.size(3)))

            # call evaluator batch_update for each metric (these update internal state or return stat)
            f1 = self.f1.batch_update(predict=pred_tensor, mask=gt_tensor)
            auc = self.auc.batch_update(predict=pred_tensor, mask=gt_tensor)
            iou = self.iou.batch_update(predict=pred_tensor, mask=gt_tensor)
            acc = self.acc.batch_update(predict=pred_tensor, mask=gt_tensor)
            fpr = self.f1.Cal_FPR(predict=pred_tensor, mask=gt_tensor)

            pred_bin = (pred_tensor > 0.5).float()
            gt_bin = (gt_tensor > 0.5).float()

            tp = (pred_bin * gt_bin).sum()
            tn = ((1 - pred_bin) * (1 - gt_bin)).sum()
            fp = (pred_bin * (1 - gt_bin)).sum()
            fn = ((1 - pred_bin) * gt_bin).sum()

            pos_acc = tp / (tp + fn + 1e-8)
            neg_acc = tn / (tn + fp + 1e-8)
            b_acc = (pos_acc + neg_acc) / 2.0

            # append scalars to lists, skipping NaNs
            if not torch.any(torch.isnan(f1)):
                total_f1.append(f1.item())
            if not torch.any(torch.isnan(auc)):
                total_auc.append(auc.item())
            if not torch.any(torch.isnan(iou)):
                total_iou.append(iou.item())
            if not torch.any(torch.isnan(acc)):
                total_acc.append(acc.item())
            if not torch.any(torch.isnan(fpr)):
                total_fpr.append(fpr.item())

            total_pos_acc.append(pos_acc.item())
            total_neg_acc.append(neg_acc.item())
            total_b_acc.append(b_acc.item())

        # compute aggregated means and print/write a summary line
        n_images = len(total_f1)
        msg = (
            f"N:{n_images}, "
            f"Tampering F1:{np.mean(total_f1):.5f}, AUC:{np.mean(total_auc):.5f}, "
            f"IoU:{np.mean(total_iou):.5f}, Acc:{np.mean(total_acc):.5f}, FPR:{np.mean(total_fpr):.5f}, "
            f"Pos_Acc:{np.mean(total_pos_acc):.5f}, Neg_Acc:{np.mean(total_neg_acc):.5f}, "
            f"B_Acc:{np.mean(total_b_acc):.5f}\n"
        )
        print(msg)

        # append to record file under save_path
        with open(os.path.join(save_path, "record.txt"), "a+") as f:
            f.write(msg)

class Evaluation_Fidelity(object):
    """
    Evaluation helper that reads original and watermarked images from disk,
    computes image-similarity metrics (PSNR/SSIM/LPIPS).
    """

    def __init__(self, wm_path, ori_path, eval_size, end_idx=None) -> None:
        self.wm_path = wm_path
        self.ori_path = ori_path
        self.eval_size = eval_size
        self.end_idx = end_idx

    def run(self, save_path):
        total_psnr = []
        total_ssim = []
        total_lpips = []
        transform = T.Compose([
            T.Resize((self.eval_size, self.eval_size)), # Bilinear
            T.ToTensor()
        ])

        wm_images_path = sorted(os.listdir(self.wm_path))
        if self.end_idx is not None:
            wm_images_path = wm_images_path[:self.end_idx]

        for wm_image_path in tqdm(wm_images_path):
            try:
                wm_image = Image.open(os.path.join(self.wm_path, wm_image_path)).convert("RGB")
                ori_image = Image.open(os.path.join(self.ori_path, wm_image_path)).convert("RGB")

                wm_tensor = transform(wm_image).unsqueeze(0)
                ori_tensor = transform(ori_image).unsqueeze(0)  

            except Exception:
                continue

            psnr_value = psnr(wm_tensor, ori_tensor).mean()
            ssim_value = ssim(wm_tensor, ori_tensor).mean()
            lpips_value = LPIPS(reduction='none')(wm_tensor, ori_tensor).mean()

            if not torch.any(torch.isnan(psnr_value)):
                total_psnr.append(psnr_value.item())
            if not torch.any(torch.isnan(ssim_value)):
                total_ssim.append(ssim_value.item())
            if not torch.any(torch.isnan(lpips_value)):
                total_lpips.append(lpips_value.item())

        n_images = len(total_psnr)
        msg = f"N:{n_images}, Tampering PSNR:{np.mean(total_psnr):.5f}, SSIM:{np.mean(total_ssim):.5f}, LPIPS: {np.mean(total_lpips):.5f}\n"
        print(msg)

        with open(os.path.join(save_path, "record.txt"), "a+") as f:
            f.write(msg)


@torch.no_grad()
def generate_watermark_image(
    norm,
    weight_path,
    target_model,
    src_image_path,
    save_path,
    edit_model_name,
    seed,
    num_bits=48,
    model_size=512,
    eval_size=256,
    start_idx=0,
    end_idx=None,
    tamper_mode='ldm',
    wm_strength=None,
):
    res = ['cover_images', 'ldm_spliced_images', 'ldm_spliceless_images', 'gt', 'msgs']

    for n in res:
        os.makedirs(os.path.join(save_path, '%s' % n), exist_ok=True)

    pipe = StableDiffusionInpaintPipeline.from_pretrained(edit_model_name, safety_checker=None).to('cuda')

    if target_model == "stableguard":
        from stableguard.models import MultiplexingWatermarkVAEDecoder
        original_vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="vae").to('cuda')
        mpw_vae_decoder = MultiplexingWatermarkVAEDecoder(num_bits=num_bits)
        mpw_vae_decoder_weight = torch.load(os.path.join(weight_path, "mpw_vae_decoder.bin"), map_location="cpu")
        mpw_vae_decoder.load_state_dict(mpw_vae_decoder_weight)

        original_vae = original_vae.cuda()
        mpw_vae_decoder = mpw_vae_decoder.cuda()
        original_vae.eval()
        mpw_vae_decoder.eval()

    elif target_model == "wam":
        from watermark_anything.wam_utils import load_model_from_checkpoint
        wam = load_model_from_checkpoint(weight_path, num_bits, scaling_w=wm_strength).cuda().eval()

    elif target_model == "omniguard":
        from omniguard.model_invert import Model, init_model
        from omniguard.modules.Unet_common import DWT
        net = Model(checkpoint=weight_path).cuda().eval()
        init_model(net)
        state_dicts = torch.load(os.path.join(weight_path, "model_checkpoint_01500.pt"), map_location="cpu", weights_only=False)
        network_state_dict = {k.removeprefix('module.'):v for k,v in state_dicts['net'].items()}
        net.load_state_dict(network_state_dict)
        print(f"OmniGuard wm_strength: {wm_strength}")

    val_dataset = AGEDataset(data_root=src_image_path, norm_type=norm, mode="val", size=eval_size)
    val_dataloader = DataLoader(
        val_dataset,
        shuffle=False,
        collate_fn=age_collate_fn,
        batch_size=1,
        num_workers=4,
        drop_last=False,
    )

    if start_idx > 0:
        print(f"Resuming from index {start_idx}...")

    for i, batch in enumerate(tqdm(val_dataloader)):

        if i < start_idx:
            continue

        if end_idx is not None and i == end_idx:
            break

        images = batch["images"].cuda()
        masks = batch["masks"].cuda()
        masks = convert_mask_to_rect(masks, scale_factor=1.2)
        image_names = batch["image_names"]

        # embed watermark
        if target_model == "stableguard":
            latents = original_vae.encode(images).latent_dist.sample()
            latents = original_vae.post_quant_conv(latents)

            phi = torch.empty(latents.size(0), num_bits).uniform_(0, 1).cuda()
            msgs = (torch.bernoulli(phi) + 1e-8)

            cover_images = mpw_vae_decoder(latents, msgs=msgs)

        elif target_model == "wam":
            images_down = F.interpolate(images, size=(model_size, model_size), mode="bilinear", align_corners=False)
            msgs = wam.get_random_msg(1)
            outputs = wam.embed(images_down, msgs)
            cover_images = outputs['imgs_w']
            cover_images = denorm_imagenet(cover_images) # [0, 1]
            cover_images = cover_images * 2.0 - 1.0 # [-1, 1]
        
        elif target_model == "omniguard":
            dwt = DWT()
            image = Image.open("./omniguard/bluesky_white2.png").convert("RGB").resize((model_size, model_size))
            result = np.array(image) / 255.
            expanded_matrix = np.expand_dims(result, axis=0) 
            secret = torch.from_numpy(np.ascontiguousarray(expanded_matrix)).float()
            secret = secret.permute(0, 3, 1, 2).cuda()

            cover_input = F.interpolate(images, size=(model_size, model_size), mode="bilinear", align_corners=False)
            secret_input = F.interpolate(secret, size=(model_size, model_size), mode="bilinear", align_corners=False)

            cover_input = dwt((cover_input + 1.0) / 2.0) # [-1, 1] to [0, 1], [1, 12, 256, 256]
            secret_input = dwt(secret_input) # [1, 12, 256, 256]
            msgs = torch.randint(2, (1, 64)).to(torch.float32).cuda()

            cover_images, output_z, out_temp, secret_temp = net(cover_input, secret_input, msgs, wm_strength=wm_strength if wm_strength is not None else 1.0)
            cover_images = cover_images * 2.0 - 1.0 # [-1, 1]

        elif target_model == "apt":
            cover_path = os.path.join(save_path, 'cover_images')
            cover_images = Image.open(os.path.join(cover_path, image_names[0])).convert("RGB").resize((model_size, model_size))
            cover_images = ToTensor()(cover_images).unsqueeze(0).cuda() # [0, 1]
            cover_images = cover_images * 2.0 - 1.0 # [-1, 1]

        generator = torch.Generator().manual_seed(seed + i)

        # tamper
        if tamper_mode == 'ldm':
            inpaint_input = F.interpolate(cover_images, size=(512, 512), mode="bilinear", align_corners=False)
            generated_images = pipe(prompt="", image=inpaint_input, mask_image=masks, generator=generator, num_inference_steps=50).images[0]

            generated_images = ToTensor()(generated_images).cuda()
            generated_images = (generated_images * 2.0 - 1.0).unsqueeze(0)

            masks = F.interpolate(masks, size=(512, 512), mode="nearest")
            spliced_images = masks * generated_images + (1 - masks) * inpaint_input # operation in [-1, 1]
            spliceless_images = generated_images

            spliced_images = F.interpolate(spliced_images, size=(model_size, model_size), mode="bilinear", align_corners=False)
            spliceless_images = F.interpolate(spliceless_images, size=(model_size, model_size), mode="bilinear", align_corners=False)
            cover_images = F.interpolate(cover_images, size=(model_size, model_size), mode="bilinear", align_corners=False)

            for i in range(images.size(0)):
                save_file_name = image_names[i]
                cover_image = cover_images[i]
                spliced_image = spliced_images[i]
                spliceless_image = spliceless_images[i]
                mask = masks[i]

                cover_image = cover_image.unsqueeze(0)
                spliced_image = spliced_image.unsqueeze(0)
                spliceless_image = spliceless_image.unsqueeze(0)
                mask = mask.unsqueeze(0)

                cover_image = (cover_image / 2 + 0.5).clamp(0, 1)
                cover_image = cover_image.squeeze(0).cpu().clamp(0, 1).numpy().transpose(1, 2, 0)
                cover_image = (cover_image * 255).astype(np.uint8)
                cover_image_pil = Image.fromarray(cover_image)

                spliced_image = (spliced_image / 2 + 0.5).clamp(0, 1)
                spliced_image = spliced_image.squeeze(0).cpu().clamp(0, 1).numpy().transpose(1, 2, 0)
                spliced_image = (spliced_image * 255).astype(np.uint8)
                spliced_image_pil = Image.fromarray(spliced_image)

                spliceless_image = (spliceless_image / 2 + 0.5).clamp(0, 1)
                spliceless_image = spliceless_image.squeeze(0).cpu().clamp(0, 1).numpy().transpose(1, 2, 0)
                spliceless_image = (spliceless_image * 255).astype(np.uint8)
                spliceless_image_pil = Image.fromarray(spliceless_image)

                if target_model != 'apt':
                    cover_image_pil.save(os.path.join(save_path, 'cover_images', save_file_name.replace("jpg", "png")))
                spliced_image_pil.save(os.path.join(save_path, 'ldm_spliced_images', save_file_name.replace("jpg", "png")))
                spliceless_image_pil.save(os.path.join(save_path, 'ldm_spliceless_images', save_file_name.replace("jpg", "png")))

                save_image(1-mask, os.path.join(save_path, 'gt', save_file_name.replace("jpg", "png")), normalize=True, scale_each=True)


@torch.no_grad()
def generate_tamper_mask(weight_path, eval_setting, target_model, save_path, num_bits=48, model_size=512, end_idx=None, wm_strength=None, use_refiner=True):
    exp_suffix = eval_setting

    refiner_tag = '_refiner' if (target_model == 'apt' and use_refiner) else ''

    # make output folder for predicted masks
    os.makedirs(os.path.join(save_path, f"pred_mask_{exp_suffix}{refiner_tag}"), exist_ok=True)
    os.makedirs(os.path.join(save_path, f"pred_bin_mask_{exp_suffix}{refiner_tag}"), exist_ok=True)
    tamper_image_path = os.path.join(save_path, f"{eval_setting}_images")

    valid_exts = (".jpg", ".jpeg", ".png")

    # initialize and load the detector model
    if target_model == "stableguard":
        from stableguard.models import MoEGuidedForensicNet
        moe_gfn = MoEGuidedForensicNet(num_bits=num_bits)
        moe_gfn_weight = torch.load(os.path.join(weight_path, "moe_gfn.bin"), map_location="cpu")
        moe_gfn.load_state_dict(moe_gfn_weight)
        moe_gfn = moe_gfn.cuda()
        moe_gfn.eval()
    
    elif target_model == "wam":
        from watermark_anything.wam_utils import load_model_from_checkpoint
        wam = load_model_from_checkpoint(weight_path, num_bits, scaling_w=wm_strength).cuda().eval()
    
    elif target_model == "omniguard":
        import albumentations as albu
        from albumentations.pytorch import ToTensorV2
        from omniguard.model_invert import Model, init_model
        from omniguard.modules.Unet_common import DWT, IWT
        from omniguard.iml_vit_model import iml_vit_model
        net = Model(checkpoint=weight_path).cuda().eval()
        init_model(net)
        state_dicts = torch.load(os.path.join(weight_path, "model_checkpoint_01500.pt"), map_location="cpu", weights_only=False)
        network_state_dict = {k.removeprefix('module.'):v for k,v in state_dicts['net'].items()}
        net.load_state_dict(network_state_dict)

        extractor = iml_vit_model()
        extractor.load_state_dict(torch.load(os.path.join(weight_path, "checkpoint-175.pth"), weights_only=False)['model'], strict=True)
        extractor = extractor.cuda().eval()

    elif target_model == "apt":
        args = Params()
        apt_model = APT(args=args)

    file_paths = os.listdir(tamper_image_path)
    image_paths = [f for f in file_paths if f.lower().endswith(valid_exts)]
    image_paths.sort()
    if end_idx is not None:
        image_paths = image_paths[:end_idx]

    for image_path in tqdm(image_paths):

        image = Image.open(os.path.join(tamper_image_path, image_path)).resize((model_size, model_size))

        if target_model == "stableguard":
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x * 2 - 1),  # map [0,1] -> [-1,1] if model expects that
            ])
            image = transform(image).unsqueeze(0)  # shape [1, C, H, W]

            pred_msgs, pred_mask = moe_gfn(image.cuda())

            pred_mask = torch.sigmoid(pred_mask)

            save_image(1-pred_mask, os.path.join(save_path, f"pred_mask_{exp_suffix}", image_path), normalize=False, scale_each=True)

        elif target_model == "wam":
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            image = transform(image).unsqueeze(0).cuda()  # shape [1, C, H, W]
            image_down = F.interpolate(image, size=(model_size, model_size), mode="bilinear", align_corners=False)

            outputs = wam.detect(image_down)["preds"]
            pred_mask = F.sigmoid(outputs[:, 0, :, :]).unsqueeze(0) # [1, 1, 256, 256]

            save_image(pred_mask, os.path.join(save_path, f"pred_mask_{exp_suffix}", image_path), normalize=False, scale_each=True)
        
        elif target_model == "omniguard":
            dwt = DWT()
            iwt = IWT()
            transform = transforms.Compose([
                transforms.Resize((model_size, model_size)),
                transforms.ToTensor(),
            ])
            transform_extractor = albu.Compose([
                albu.PadIfNeeded(          
                    min_height=1024,
                    min_width=1024, 
                    border_mode=0, 
                    value=0, 
                    position='top_left',
                    mask_value=0),
                albu.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                albu.Crop(0, 0, 1024, 1024),
                ToTensorV2()
            ])
            image = transform(image).unsqueeze(0)
            output_steg = dwt(image)
            output_image, bits = net(output_steg.cuda(), rev=True)
            secret_rev = output_image.narrow(1, 0, 12)
            secret_rev = iwt(secret_rev)

            artifact = secret_rev.permute(0, 2, 3, 1).squeeze().cpu().numpy() * 255
            fuse = image.permute(0, 2, 3, 1).squeeze().cpu().numpy() * 255

            artifact = transform_extractor(image=artifact)['image'].cuda().unsqueeze(0)
            fuse = transform_extractor(image=fuse)['image'].cuda().unsqueeze(0)

            pred_mask = extractor(artifact, fuse)
            pred_mask = pred_mask[:, :, 0:model_size, 0:model_size]
            pred_mask = F.interpolate(pred_mask, size=(model_size, model_size), mode="bilinear", align_corners=False)

            save_image(1-pred_mask, os.path.join(save_path, f"pred_mask_{exp_suffix}", image_path), normalize=False, scale_each=True)

        elif target_model == "apt":
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            image = transform(image).unsqueeze(0)

            logits, pred_mask, bin_prediction = apt_model.decode_watermark(image.cuda(), use_refiner=use_refiner)

            save_image(pred_mask, os.path.join(save_path, f"pred_mask_{exp_suffix}{refiner_tag}", image_path), normalize=False, scale_each=False)
            save_image(bin_prediction, os.path.join(save_path, f"pred_bin_mask_{exp_suffix}{refiner_tag}", image_path), normalize=False, scale_each=False)

def save_and_print_config(config, save_path):
    """
    Save and print the configuration dictionary to a YAML file.
    """
    os.makedirs(save_path, exist_ok=True)
    
    config_path = os.path.join(save_path, "config_eval.yaml")

    print("-" * 30)
    print(" " * 10 + "Configuration")
    print("-" * 30)
    
    print(yaml.dump(config, allow_unicode=True, default_flow_style=False))
    
    print("-" * 30)

    with open(config_path, 'w') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
        
    print(f"Configuration saved to: {config_path}")
    print("-" * 30)

if __name__ == "__main__":
    base_conf = OmegaConf.load('config.yaml')
    cli_conf = OmegaConf.from_cli()
    final_conf = OmegaConf.merge(base_conf, cli_conf)
    c = OmegaConf.to_container(final_conf, resolve=True)
    
    # model-specific settings
    model_name = c['target_model']
    c['weight_path'] = c['weight_paths'][model_name]
    c['normalization'] = c['normalization'][model_name]
    c['num_bits'] = c['num_bits'][model_name]
    c['model_size'] = c['train_img_size'][model_name]
    c['wm_strength'] = c.get('wm_strength', None)  # WAM/OmniGuard watermark strength
    c['use_refiner'] = c.get('use_refiner', True)  # apt: use mask_refiner or direct upsampling

    # evaluation settings based on tamper mode
    eval_setting = []
    if c['tamper_mode'] == 'ldm':
        eval_setting = ["ldm_spliced", "ldm_spliceless"]

    print("-" * 30)
    print("Running Configuration:")
    print(OmegaConf.to_yaml(final_conf))
    print("-" * 30)

    set_seed(c['seed'])
    # 1) generate watermarked/ tampered images and save cover/tamper/gt/msg to disk
    # save_and_print_cfg = save_and_print_config(c, c['save_path'])
    generate_watermark_image(norm=c['normalization'],
                             weight_path=c['weight_path'],
                             target_model=c['target_model'],
                             src_image_path=c['src_image_path'],
                             save_path=c['save_path'],
                             edit_model_name=c['edit_model_name'],
                             seed=c['seed'],
                             num_bits=c['num_bits'],
                             model_size=c['model_size'],
                             eval_size=c['eval_size'],
                             start_idx=c['start_idx'],
                             end_idx=c['end_idx'],
                             tamper_mode=c['tamper_mode'],
                             wm_strength=c['wm_strength'])

    # # 2) run detector over the saved spliced/spliceless images to generate predicted masks and message predictions
    refiner_tag = '_refiner' if (c['target_model'] == 'apt' and c['use_refiner']) else ''
    for setting in eval_setting:
        generate_tamper_mask(weight_path=c['weight_path'],
                            eval_setting=setting,
                            target_model=c['target_model'],
                            save_path=c['save_path'],
                            num_bits=c['num_bits'],
                            model_size=c['model_size'],
                            end_idx=c['end_idx'],
                            wm_strength=c['wm_strength'],
                            use_refiner=c['use_refiner'])
        # 3) Evaluate predicted masks against ground-truth masks saved in disk
        pred_mask_dir = f"{c['save_path']}/pred_mask_{setting}{refiner_tag}"
        eva = Evaluation(pred_mask_dir, f"{c['save_path']}/gt", eval_size=c['eval_size'], end_idx=c['end_idx'])
        eva.run(pred_mask_dir)

    # # 4) Evaluate fidelity between watermarked and original images
    # eva_fid = Evaluation_Fidelity(f"{c['save_path']}/cover_images", f"{c['src_image_path']}", eval_size=c['eval_size'], end_idx=c['end_idx'])
    # eva_fid.run(f"{c['save_path']}/cover_images")

