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
from ct_clip import CTCLIP

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Every volume is reoriented to these axis codes before any array indexing.
# nii_to_tensor() assumes axis 2 is the superior/inferior (slice) axis and that
# axes 0/1 are in-plane; that only holds after an explicit reorientation, since
# get_fdata() returns voxels in whatever order the file happens to store them.
TARGET_AXCODES = ("L", "P", "S")


def reorient_to_target(img, axcodes=TARGET_AXCODES):
    """
    Permute and flip a NIfTI image so that nib.aff2axcodes(img.affine) == axcodes.

    Returns the reoriented image; both the array and the affine are updated, so
    voxel spacing must be re-read from the returned affine (the original header's
    pixdim no longer maps to the new (i, j, k) order).
    """
    start = nib.io_orientation(img.affine)
    end = nib.orientations.axcodes2ornt(axcodes)
    reoriented = img.as_reoriented(nib.orientations.ornt_transform(start, end))

    got = nib.aff2axcodes(reoriented.affine)
    assert tuple(got) == tuple(axcodes), (
        f"Reorientation failed: got axis codes {got}, expected {tuple(axcodes)}"
    )
    return reoriented


def resize_array(array, current_spacing, target_spacing):
    """
    Resize the array to match the target spacing.

    Args:
    array (torch.Tensor): Input array to be resized.
    current_spacing (tuple): Current voxel spacing (z_spacing, xy_spacing, xy_spacing).
    target_spacing (tuple): Target voxel spacing (target_z_spacing, target_x_spacing, target_y_spacing).

    Returns:
    np.ndarray: Resized array.
    """
    # Calculate new dimensions
    original_shape = array.shape[2:]
    scaling_factors = [
        current_spacing[i] / target_spacing[i] for i in range(len(original_shape))
    ]
    new_shape = [
        int(original_shape[i] * scaling_factors[i]) for i in range(len(original_shape))
    ]
    # Resize the array
    resized_array = F.interpolate(array, size=new_shape, mode='trilinear', align_corners=False).cpu().numpy()
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
      1. Load NIfTI, reorient to TARGET_AXCODES (LPS), read spacing from the
         reoriented affine
      2. Apply HU clipping [-1000, 1000]
      3. Resample to target spacing (1.5mm Z, 0.75mm XY)
      4. Normalize by dividing by 1000 -> [-1, 1]
      5. Crop/pad to 480x480x240 around mask ROI (or image center if no mask)
      6. Permute to (D, H, W) and add channel dim -> (1, 1, D, H, W)

    If mask_path is provided, the mask is resampled alongside the image and its
    centroid drives the crop, matching 3DINO's MaskCenterCropd (with foreground
    label filtering and a fallback to the original mask). Without a mask, the
    crop/pad is the verbatim center-crop block from scripts/data.py.
    """
    nii_img = nib.load(str(path))

    # Standardize orientation to TARGET_AXCODES before touching the array, so axis 2 is always superior/inferior and axes 0/1 are always in-plane.
    nii_img = reorient_to_target(nii_img)
    img_data = nii_img.get_fdata().astype(np.float32)
    # nibabel's get_fdata() already applies scl_slope / scl_inter; do not re-apply.

    # Spacing comes from the reoriented affine, not the original header pixdim:
    # Axes are now (L, P, S).
    l_spacing, p_spacing, s_spacing = (
        float(v) for v in nib.affines.voxel_sizes(nii_img.affine)
    )

    mask_data_orig = None
    if mask_path is not None:
        # The mask is reoriented to the same axis codes, so image and mask voxels stay in correspondence.
        mask_nii = reorient_to_target(nib.load(str(mask_path)))
        mask_data_orig = mask_nii.get_fdata().astype(np.float32)
        assert mask_data_orig.shape == img_data.shape, (
            f"Image/mask shape mismatch: {img_data.shape} vs {mask_data_orig.shape}"
        )

    # (L, P, S) -> (S, L, P) = (D, H, W) to match CT-CLIP convention
    img_data = img_data.transpose(2, 0, 1)
    if mask_data_orig is not None:
        mask_data_orig_dhw = mask_data_orig.transpose(2, 0, 1)

    current = (s_spacing, l_spacing, p_spacing)
    target = (1.5, 0.75, 0.75)

    img_data = np.clip(img_data, -1000, 1000)

    img_tensor = torch.tensor(img_data).unsqueeze(0).unsqueeze(0).float()
    img_data = resize_array(img_tensor, current, target)[0][0]

    mask_resampled = None
    if mask_data_orig is not None:
        new_shape = img_data.shape  # (D, H, W) after resampling
        mask_tensor = torch.tensor(mask_data_orig_dhw).unsqueeze(0).unsqueeze(0).float()
        mask_tensor = F.interpolate(mask_tensor, size=new_shape, mode='nearest')
        mask_resampled = mask_tensor[0, 0].numpy()

    # Normalize: divide by 1000 -> [-1, 1]
    img_data = (img_data / 1000.0).astype(np.float32)

    if mask_resampled is not None:
        # Crop to (D=240, H=480, W=480) around the mask ROI centroid.
        center = mask_center_from_resampled(
            mask_resampled, mask_data_orig_dhw, fg_labels=fg_labels
        )
        img_data = crop_with_padding(img_data, center, (240, 480, 480), pad_value=-1.0)
        tensor = torch.tensor(img_data)
    else:
        # works in (H, W, D) order and permutes back to (D, H, W) at the end.
        img_data = np.transpose(img_data, (1, 2, 0))

        tensor = torch.tensor(img_data)
        # Get the dimensions of the input tensor
        target_shape = (480, 480, 240)

        # Extract dimensions
        h, w, d = tensor.shape

        # Calculate cropping/padding values for height, width, and depth
        dh, dw, dd = target_shape
        h_start = max((h - dh) // 2, 0)
        h_end = min(h_start + dh, h)
        w_start = max((w - dw) // 2, 0)
        w_end = min(w_start + dw, w)
        d_start = max((d - dd) // 2, 0)
        d_end = min(d_start + dd, d)

        # Crop or pad the tensor
        tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]

        pad_h_before = (dh - tensor.size(0)) // 2
        pad_h_after = dh - tensor.size(0) - pad_h_before

        pad_w_before = (dw - tensor.size(1)) // 2
        pad_w_after = dw - tensor.size(1) - pad_w_before

        pad_d_before = (dd - tensor.size(2)) // 2
        pad_d_after = dd - tensor.size(2) - pad_d_before

        tensor = torch.nn.functional.pad(tensor, (pad_d_before, pad_d_after, pad_w_before, pad_w_after, pad_h_before, pad_h_after), value=-1)

        tensor = tensor.permute(2, 0, 1)

    # (D, H, W) -> (batch=1, channels=1, D, H, W)
    tensor = tensor.unsqueeze(0).unsqueeze(0)

    return tensor


def build_ctclip_model(checkpoint_path):
    """
    Build and load the CT-CLIP model following the ClassFine/LiPro pattern.
    Returns the CLIP model.

    Built with image_only=True: extraction only ever calls clip.forward_image(),
    so no text encoder or tokenizer is constructed and nothing here touches
    HuggingFace. clip.load() drops the checkpoint's text_transformer.* keys to
    match.
    """
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
        image_only=True,
        dim_image=294912,
        dim_text=768,
        dim_latent=512,
        extra_latent_projection=False,
        use_mlm=False,
        downsample_image_embeds=False,
        use_all_token_embeds=False,
    )

    clip.load(checkpoint_path)
    return clip


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
    clip = build_ctclip_model(args.checkpoint)
    clip.eval()
    clip.to(device)

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

            # Image-only forward pass: identical to the image branch of clip(..., return_latents=True), but encodes no text at all.
            # Returns: (image_latents, enc_image_send)
            image_latents, _ = clip.forward_image(image_tensor)

            # --- ReLU on latents (matches scripts/ct_lipro_train.py's
            # ImageLatentsClassifier.forward, minus the dropout, which is
            # train-only anyway). Comment out these 2 lines to disable.
            relu = torch.nn.ReLU()
            image_latents = relu(image_latents)
            # --- end ReLU

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
