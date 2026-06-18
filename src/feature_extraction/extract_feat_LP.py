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


def crop_with_padding(arr, center, size, pad_value=-1.0):
    """
    Crop a 3D array (D, H, W) around `center` to `size`, zero/pad as needed.
    Ported from 3DINO's MaskCenterCropd._crop_with_padding.
    """
    zc, yc, xc = center
    dz_b, dy_b, dx_b = size[0] // 2, size[1] // 2, size[2] // 2
    dz_a, dy_a, dx_a = size[0] - dz_b, size[1] - dy_b, size[2] - dx_b

    z_start, z_end = zc - dz_b, zc + dz_a
    y_start, y_end = yc - dy_b, yc + dy_a
    x_start, x_end = xc - dx_b, xc + dx_a

    cropped = np.full(size, pad_value, dtype=arr.dtype)

    z_sv, y_sv, x_sv = max(z_start, 0), max(y_start, 0), max(x_start, 0)
    z_ev = min(z_end, arr.shape[0])
    y_ev = min(y_end, arr.shape[1])
    x_ev = min(x_end, arr.shape[2])

    z_off, y_off, x_off = z_sv - z_start, y_sv - y_start, x_sv - x_start

    cropped[
        z_off:z_off + (z_ev - z_sv),
        y_off:y_off + (y_ev - y_sv),
        x_off:x_off + (x_ev - x_sv),
    ] = arr[z_sv:z_ev, y_sv:y_ev, x_sv:x_ev]

    return cropped


def mask_center_from_resampled(mask_resampled, mask_original, fg_labels=None):
    """
    Find ROI center in the resampled (D, H, W) frame.
    Falls back to scaling coords from the original mask if resampled is empty,
    and finally to the image center. Mirrors 3DINO MaskCenterCropd behavior.
    """
    if fg_labels is not None:
        mask_bin = np.isin(mask_resampled, fg_labels)
    else:
        mask_bin = mask_resampled > 0

    coords = np.argwhere(mask_bin)
    if coords.size > 0:
        return tuple(coords.mean(axis=0).astype(int))

    # Fall back to the un-resampled mask, rescaling coords into the resampled frame
    if fg_labels is not None:
        mask_orig_bin = np.isin(mask_original, fg_labels)
    else:
        mask_orig_bin = mask_original > 0
    coords_orig = np.argwhere(mask_orig_bin)

    shape_r = mask_resampled.shape
    if coords_orig.size > 0:
        shape_o = mask_original.shape
        scale = np.array([shape_r[i] / shape_o[i] for i in range(3)])
        scaled = (coords_orig * scale).astype(int)
        return tuple(scaled.mean(axis=0).astype(int))

    print("No foreground voxels in mask (resampled or original); using image center.")
    return (shape_r[0] // 2, shape_r[1] // 2, shape_r[2] // 2)


def nii_to_tensor(path, mask_path=None, fg_labels=None):
    """
    Load a NIfTI file and preprocess it for CT-CLIP.
    Preprocessing follows CT-CLIP's data_inference_nii.py pipeline:
      1. Load NIfTI and get spacing from header
      2. Apply HU clipping [-1000, 1000]
      3. Resample to target spacing (1.5mm Z, 0.75mm XY)
      4. Normalize by dividing by 1000 -> [-1, 1]
      5. Crop/pad to 480x480x240 around mask ROI (or image center if no mask)
      6. Permute to (D, H, W) and add channel dim -> (1, 1, D, H, W)

    If mask_path is provided, the mask is resampled alongside the image and its
    centroid drives the crop, matching 3DINO's MaskCenterCropd (with foreground
    label filtering and a fallback to the original mask).
    """
    nii_img = nib.load(str(path))
    img_data = nii_img.get_fdata().astype(np.float32)

    header = nii_img.header
    pixdim = header['pixdim']
    xy_spacing = float(pixdim[1])
    z_spacing = float(pixdim[3])
    # nibabel's get_fdata() already applies scl_slope / scl_inter; do not re-apply.

    mask_data_orig = None
    if mask_path is not None:
        mask_nii = nib.load(str(mask_path))
        mask_data_orig = mask_nii.get_fdata().astype(np.float32)
        assert mask_data_orig.shape == img_data.shape, (
            f"Image/mask shape mismatch: {img_data.shape} vs {mask_data_orig.shape}"
        )

    # HU clipping
    img_data = np.clip(img_data, -1000, 1000)

    # (X, Y, Z) -> (Z, X, Y) = (D, H, W) to match CT-CLIP convention
    img_data = img_data.transpose(2, 0, 1)
    if mask_data_orig is not None:
        mask_data_orig_dhw = mask_data_orig.transpose(2, 0, 1)

    # Resample to target spacing (image: trilinear; mask: nearest)
    current = (z_spacing, xy_spacing, xy_spacing)
    target = (1.5, 0.75, 0.75)

    img_tensor = torch.tensor(img_data).unsqueeze(0).unsqueeze(0).float()
    img_tensor = resize_array(img_tensor, current, target)
    img_data = img_tensor[0, 0].numpy()

    mask_resampled = None
    if mask_data_orig is not None:
        new_shape = img_data.shape  # (D, H, W) after resampling
        mask_tensor = torch.tensor(mask_data_orig_dhw).unsqueeze(0).unsqueeze(0).float()
        mask_tensor = F.interpolate(mask_tensor, size=new_shape, mode='nearest')
        mask_resampled = mask_tensor[0, 0].numpy()

    # Normalize: divide by 1000 -> [-1, 1]
    img_data = (img_data / 1000.0).astype(np.float32)

    # Crop to (D=240, H=480, W=480) around ROI center (or image center)
    target_shape_dhw = (240, 480, 480)
    if mask_resampled is not None:
        center = mask_center_from_resampled(
            mask_resampled, mask_data_orig_dhw, fg_labels=fg_labels
        )
    else:
        center = tuple(s // 2 for s in img_data.shape)

    img_data = crop_with_padding(img_data, center, target_shape_dhw, pad_value=-1.0)

    tensor = torch.tensor(img_data)
    # (D, H, W) -> (batch=1, channels=1, D, H, W)
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
    ap.add_argument("--fg_labels", type=int, nargs='+', default=None,
                    help='Foreground mask label(s) to use for ROI center '
                         '(e.g. --fg_labels 1). Defaults to all non-zero voxels.')

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
                image_tensor = nii_to_tensor(
                    img_full_path, mask_path=mask_full_path, fg_labels=args.fg_labels
                )
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
            # assert path does not exist already
            assert not os.path.exists(single_out_path)
            with h5py.File(single_out_path, 'w') as hf:
                hf.create_dataset('y_hat', data=image_latents[0].numpy())

            processed_count += 1

            # Clean up
            del image_tensor, image_latents
            torch.cuda.empty_cache()

    print(f"Done. Processed {processed_count}/{len(imgs_files)} images.")
