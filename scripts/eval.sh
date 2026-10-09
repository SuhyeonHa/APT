#!/bin/bash

set -e

NUM_IMG=100
TAMPER_MODEL=ldm
DATA=./data/valAGE-Set
OUT=./runs/eval

# ============================ APT / APT* (ours) ============================
# Two stages: (1) optimize the watermark (writes cover images), then
# (2) tamper + localize + score. The APT branch of eval_AGE.py LOADS the cover
# images from save_path/cover_images, so save_path must be the embedding output.

# Stage 1 — optimize watermarks for the first NUM_IMG images.
# Writes runs/apt/apt/<timestamp>/cover_images (see apt/main.py Params).
python -m apt.main
EXP=$(ls -td ./runs/apt/apt/*/ | head -1)
echo "APT cover images: $EXP/cover_images"

# Stage 2 — evaluate. use_refiner=True -> APT*, False -> APT.
for REF in True False; do
  python eval_AGE.py \
      target_model=apt \
      src_image_path=$DATA \
      save_path="$EXP" \
      edit_model_name=sd-legacy/stable-diffusion-inpainting \
      eval_size=256 \
      start_idx=0 \
      end_idx=$NUM_IMG \
      tamper_mode=$TAMPER_MODEL \
      use_refiner=$REF
done

# ===========================================================================

# ---- WAM baseline ----
# python eval_AGE.py \
#     target_model=wam \
#     src_image_path=$DATA \
#     save_path=$OUT/wam \
#     edit_model_name=sd-legacy/stable-diffusion-inpainting \
#     eval_size=256 \
#     start_idx=0 \
#     end_idx=$NUM_IMG \
#     tamper_mode=$TAMPER_MODEL \
#     wm_strength=3.0

# ---- OmniGuard baseline ----
# python eval_AGE.py \
#     target_model=omniguard \
#     src_image_path=$DATA \
#     save_path=$OUT/omniguard \
#     edit_model_name=sd-legacy/stable-diffusion-inpainting \
#     eval_size=256 \
#     start_idx=0 \
#     end_idx=$NUM_IMG \
#     tamper_mode=$TAMPER_MODEL \
#     wm_strength=2.0

# ---- StableGuard baseline ----
# python eval_AGE.py \
#     target_model=stableguard \
#     src_image_path=$DATA \
#     save_path=$OUT/stableguard \
#     edit_model_name=sd-legacy/stable-diffusion-inpainting \
#     eval_size=256 \
#     start_idx=0 \
#     end_idx=$NUM_IMG \
#     tamper_mode=$TAMPER_MODEL
