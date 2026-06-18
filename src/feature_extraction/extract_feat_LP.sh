#!/bin/bash
set -e

# Default paths for Docker environment
INPUT_DIR="${INPUT_DIR:-/workspace/inputs}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/outputs}"
MASKS_DIR="${MASKS_DIR:-}"  # Optional masks directory
CHECKPOINT="${CHECKPOINT:-./checkpoints/CT-CLIP_v2.pt}"

# Build command with optional masks_path
CMD="python3 extract_feat_LP.py -i \"$INPUT_DIR\" -o \"$OUTPUT_DIR\" --checkpoint \"$CHECKPOINT\""

# Add masks_path argument if MASKS_DIR is set and not empty
if [ -n "$MASKS_DIR" ]; then
    CMD="$CMD --masks_path \"$MASKS_DIR\""
fi

# Run feature extraction
eval $CMD
