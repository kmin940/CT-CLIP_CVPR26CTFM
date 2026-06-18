import warnings
warnings.filterwarnings("ignore")

import os
import torch
import torch.nn.functional as F
import numpy as np
import nibabel as nib
import argparse
import h5py
from tqdm import tqdm

from transformer_maskgit import CTViT
from transformers import BertTokenizer, BertModel
from ct_clip import CTCLIP

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resize_array(array, current_spacing, target_spacing):
    """
    Resize the array to match the target spacing using trilinear interpolation.
    Adapted from CT-CLIP's data_inference_nii.py.
    """
    original_shape = array.shape[2:]
    scaling_factors = [
        current_spacing[i] / target_spacing[i] for i in range(len(original_shape))
    ]
    new_shape = [
        int(original_shape[i] * scaling_factors[i]) for i in range(len(original_shape))
    ]
    resized_array = F.interpolate(array, size=new_shape, mode='trilinear', align_corners=False)
    return resized_array


def center_crop_or_pad(tensor, target_shape):
    """
    Center crop or pad tensor to target_shape.
    Input tensor shape: (H, W, D). Pads with -1 (matching CT-CLIP convention).
    """
    h, w, d = tensor.shape
    dh, dw, dd = target_shape

    # Center crop
    h_start = max((h - dh) // 2, 0)
    h_end = min(h_start + dh, h)
    w_start = max((w - dw) // 2, 0)
    w_end = min(w_start + dw, w)
    d_start = max((d - dd) // 2, 0)
    d_end = min(d_start + dd, d)

    tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]

    # Pad if needed
    pad_h_before = (dh - tensor.size(0)) // 2
    pad_h_after = dh - tensor.size(0) - pad_h_before
    pad_w_before = (dw - tensor.size(1)) // 2
    pad_w_after = dw - tensor.size(1) - pad_w_before
    pad_d_before = (dd - tensor.size(2)) // 2
    pad_d_after = dd - tensor.size(2) - pad_d_before

    tensor = F.pad(tensor, (pad_d_before, pad_d_after, pad_w_before, pad_w_after, pad_h_before, pad_h_after), value=-1)
    return tensor


def nii_to_tensor(path, mask_path=None):
    """
    Load a NIfTI file and preprocess it for CT-CLIP.
    Preprocessing follows CT-CLIP's data_inference_nii.py pipeline:
      1. Load NIfTI and get spacing from header
      2. Apply HU clipping [-1000, 1000]
      3. Resample to target spacing (1.5mm Z, 0.75mm XY)
      4. Normalize by dividing by 1000 -> [-1, 1]
      5. Center crop/pad to 480x480x240
      6. Permute to (D, H, W) and add channel dim -> (1, D, H, W)

    If mask_path is provided, the mask is used for center cropping around the ROI
    before resampling.
    """
    nii_img = nib.load(str(path))
    img_data = nii_img.get_fdata().astype(np.float32)

    # Get spacing from NIfTI header
    header = nii_img.header
    pixdim = header['pixdim']
    # pixdim[1:4] gives (x, y, z) spacing
    xy_spacing = float(pixdim[1])
    z_spacing = float(pixdim[3])

    # For NIfTI files, slope/intercept are usually in the header
    slope = float(header.get('scl_slope', 1.0))
    intercept = float(header.get('scl_inter', 0.0))
    # nibabel already applies slope/intercept in get_fdata(), so skip if default
    if np.isnan(slope) or slope == 0:
        slope = 1.0
    if np.isnan(intercept):
        intercept = 0.0
    # Note: nibabel's get_fdata() already applies scl_slope and scl_inter,
    # so we do NOT re-apply them here.

    # If mask provided, do mask-centered crop before resampling
    if mask_path is not None:
        mask_nii = nib.load(str(mask_path))
        mask_data = mask_nii.get_fdata()
        # Find bounding box of mask
        nonzero = np.argwhere(mask_data > 0)
        if len(nonzero) > 0:
            mins = nonzero.min(axis=0)
            maxs = nonzero.max(axis=0)
            center = (mins + maxs) // 2
            # Crop around center with generous margin
            margin = np.array([80, 80, 40])  # in voxels, before resampling
            crop_min = np.maximum(center - margin, 0)
            crop_max = np.minimum(center + margin, np.array(img_data.shape))
            img_data = img_data[crop_min[0]:crop_max[0], crop_min[1]:crop_max[1], crop_min[2]:crop_max[2]]

    # HU clipping
    hu_min, hu_max = -1000, 1000
    img_data = np.clip(img_data, hu_min, hu_max)

    # Transpose to (D, H, W) for resampling - matching CT-CLIP convention
    img_data = img_data.transpose(2, 0, 1)

    # Resample to target spacing
    target_z_spacing = 1.5
    target_x_spacing = 0.75
    target_y_spacing = 0.75
    current = (z_spacing, xy_spacing, xy_spacing)
    target = (target_z_spacing, target_x_spacing, target_y_spacing)

    tensor = torch.tensor(img_data).unsqueeze(0).unsqueeze(0).float()
    tensor = resize_array(tensor, current, target)
    img_data = tensor[0][0].numpy()

    # Back to (H, W, D) for center crop/pad
    img_data = np.transpose(img_data, (1, 2, 0))

    # Normalize: divide by 1000 -> [-1, 1]
    img_data = (img_data / 1000.0).astype(np.float32)

    tensor = torch.tensor(img_data)

    # Center crop/pad to target shape (H=480, W=480, D=240)
    target_shape = (480, 480, 240)
    tensor = center_crop_or_pad(tensor, target_shape)

    # Permute to (D, H, W) and build 5D tensor -> (batch=1, channels=1, frames=D, H, W)
    tensor = tensor.permute(2, 0, 1)
    tensor = tensor.unsqueeze(0).unsqueeze(0)

    return tensor


def build_ctclip_model(checkpoint_path):
    """
    Build and load the CT-CLIP model following the ClassFine/LiPro pattern.
    Returns the CLIP model and tokenizer.
    """
    tokenizer = BertTokenizer.from_pretrained(
        'microsoft/BiomedVLP-CXR-BERT-specialized', do_lower_case=True
    )
    text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")
    text_encoder.resize_token_embeddings(len(tokenizer))

    image_encoder = CTViT(
        dim=512,
        codebook_size=8192,
        image_size=480,
        patch_size=20,
        temporal_patch_size=10,
        spatial_depth=4,
        temporal_depth=4,
        dim_head=32,
        heads=8,
    )

    clip = CTCLIP(
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        dim_image=294912,
        dim_text=768,
        dim_latent=512,
        extra_latent_projection=False,
        use_mlm=False,
        downsample_image_embeds=False,
        use_all_token_embeds=False,
    )

    clip.load(checkpoint_path)
    return clip, tokenizer


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    # Docker-compatible arguments
    ap.add_argument("-i", "--input", "--imgs_path", dest="imgs_path", type=str,
                    default='/workspace/inputs',
                    help='Path to input images directory')
    ap.add_argument("-o", "--output", "--dest", dest="dest", type=str,
                    default='/workspace/outputs',
                    help='Destination folder to save features')
    ap.add_argument("--masks_path", type=str, default=None,
                    help='Path to foreground masks for roi-disease (set to None for non-roi diseases)')
    ap.add_argument("--checkpoint", type=str,
                    default='./checkpoints/CT-CLIP_v2.pt',
                    help='Path to CT-CLIP checkpoint')
    ap.add_argument("--batch_size", type=int, default=1,
                    help='Batch size for feature extraction')

    args = ap.parse_args()

    # Build and load CT-CLIP model
    clip, tokenizer = build_ctclip_model(args.checkpoint)
    clip.eval()
    clip.to(device)

    # Prepare dummy text tokens (following ClassFine/LiPro pattern)
    # The text encoder receives a blank prompt; only image latents are used.
    dummy_text = tokenizer(
        [" "], return_tensors="pt", padding="max_length",
        truncation=True, max_length=512
    ).to(device)

    imgs_path = args.imgs_path
    os.makedirs(args.dest, exist_ok=True)

    # Collect input files
    imgs_files = sorted([f for f in os.listdir(imgs_path) if f.endswith('.nii.gz')])
    if args.masks_path:
        imgs_files = [f for f in imgs_files if os.path.exists(os.path.join(args.masks_path, f))]

    # Extract features
    processed_count = 0
    with torch.no_grad():
        for img_file in tqdm(imgs_files, desc="Extracting features"):
            img_id = img_file.replace('.nii.gz', '')
            img_full_path = os.path.join(imgs_path, img_file)

            mask_full_path = None
            if args.masks_path is not None:
                mask_full_path = os.path.join(args.masks_path, img_file)
                assert os.path.exists(mask_full_path), f'Mask file not found: {mask_full_path}'

            # Preprocess image using CT-CLIP pipeline
            try:
                image_tensor = nii_to_tensor(img_full_path, mask_path=mask_full_path)
            except Exception as e:
                print(f"Error preprocessing {img_file}: {e}")
                continue

            image_tensor = image_tensor.to(device)

            # Expand dummy text to match batch size
            batch_text = dummy_text

            # Forward pass through CT-CLIP with return_latents=True
            # Returns: (text_latents, image_latents, enc_image_send)
            _, image_latents, _ = clip(
                batch_text, image_tensor, device=device, return_latents=True
            )

            # image_latents shape: (1, 512) - the CLIP-projected image features
            image_latents = image_latents.detach().cpu()

            # Save h5 file
            single_out_path = os.path.join(args.dest, f'{img_id}.h5')
            with h5py.File(single_out_path, 'w') as hf:
                hf.create_dataset('y_hat', data=image_latents[0].numpy())

            processed_count += 1

            # Clean up
            del image_tensor, image_latents
            torch.cuda.empty_cache()

    print(f"Done. Processed {processed_count}/{len(imgs_files)} images.")
