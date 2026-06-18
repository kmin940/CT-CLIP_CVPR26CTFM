#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=1

DATA_ROOT="/home/jma/Documents/cryoSumin/CT_FM/data/raw_data_classify/amos-clf-tr-val/test"
INPUT_DIR="${INPUT_DIR:-${DATA_ROOT}/images}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/jma/Documents/cryoSumin/CT_FM/data/embeddings/features_LP_CT-CLIP}"
CHECKPOINT="${CHECKPOINT:-/home/jma/Documents/cryoSumin/CT_FM/CT-CLIP/checkpoints/CT-CLIP_v2.pt}"

disease_list=(
  splenomegaly
  adrenal_hyperplasia
  fatty_liver
  cholecystitis
  liver_calcifications
  hydronephrosis
  gallstone
  liver_lesion
  kidney_stone
  liver_cyst
  renal_cyst
  atherosclerosis
  colorectal_cancer
  ascites
  lymphadenopathy
)
non_roi_disease_list=(
  atherosclerosis
  colorectal_cancer
  ascites
  lymphadenopathy
)
for disease in "${disease_list[@]}"; do
    if [[ " ${non_roi_disease_list[@]} " =~ " ${disease} " ]]; then
      echo "Running feature extraction for non-roi ${disease} ..."
      MASKS_DIR=""
    else
      echo "Running feature extraction for roi ${disease} ..."
      MASKS_DIR="${DATA_ROOT}/fg_masks/${disease}"
    fi

    # Build command with optional masks_path
    CMD="python3 extract_feat_LP.py -i \"$INPUT_DIR\" -o \"$OUTPUT_DIR/${disease}/embeddings\" --checkpoint \"$CHECKPOINT\""

    # Add masks_path argument if MASKS_DIR is set and not empty
    if [ -n "$MASKS_DIR" ]; then
        CMD="$CMD --batch_size 1 --masks_path \"$MASKS_DIR\""
    else
        CMD="$CMD --batch_size 1"
    fi

    # Run feature extraction
    eval $CMD
done
