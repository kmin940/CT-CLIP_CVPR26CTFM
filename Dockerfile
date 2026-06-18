FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

# Install system dependencies
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt-get update && apt-get install -y \
    git ffmpeg libsm6 libxext6 tzdata \
 && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
 && echo $TZ > /etc/timezone \
 && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/inputs /workspace/outputs \
    && chown user:user /workspace/inputs /workspace/outputs

USER user

ENV PATH="/home/user/.local/bin:${PATH}"

# Upgrade pip
RUN python -m pip install --user -U pip && python -m pip install --user pip-tools

# Copy the CT-CLIP repository (slimmed via .dockerignore: no .git / .venv).
# This includes:
#   - checkpoints/CT-CLIP_v2.pt   (the CT-CLIP image+text weights)
#   - hf_cache/                   (the baked CXR-BERT text encoder, see below)
COPY --chown=user:user . /opt/app/

# Set working directory for installation
WORKDIR /opt/app/

# Install transformer_maskgit package (CTViT image encoder)
RUN pip install --user -e ./transformer_maskgit/

# Install CT_CLIP package (CLIP model)
RUN pip install --user -e ./CT_CLIP/

# Install additional dependencies for feature extraction.
# transformers is pinned to 4.44.2: the base image ships torch 2.1.0, and
# transformers >=5.0 hard-requires torch >=2.4 (it otherwise disables PyTorch and
# BertModel becomes unavailable). 4.44.2 loads the baked CXR-BERT text encoder on
# torch 2.1.0 and auto-resolves a compatible huggingface_hub (<1.0) / tokenizers.
RUN pip install --user nibabel h5py tqdm "transformers==4.44.2"

# Pin numpy<2 LAST. The base image's torch 2.1.0 is compiled against NumPy 1.x
# and crashes with "RuntimeError: Numpy is not available" under numpy 2.x, which
# the (unpinned) package dependencies otherwise pull in. This downgrade must be
# the final install step so nothing re-upgrades numpy.
RUN pip install --user "numpy<2"

# --- Offline text encoder -----------------------------------------------------
# CT-CLIP's text branch loads microsoft/BiomedVLP-CXR-BERT-specialized via
# transformers.from_pretrained(). The weights are baked into the image under
# hf_cache/ (staged from the host HF cache before build) so extraction needs
# NO network access at run time -- required for the offline eval container.
ENV HF_HOME=/opt/app/hf_cache
ENV TRANSFORMERS_OFFLINE=1
ENV HF_HUB_OFFLINE=1

# --- Checkpoint path ----------------------------------------------------------
# extract_feat_LP.sh honours ${CHECKPOINT:-...}; pin it to the absolute baked
# path so it resolves regardless of the working directory. (The previous default
# './checkpoints/CT-CLIP_v2.pt' was relative to this WORKDIR and did not exist.)
ENV CHECKPOINT=/opt/app/checkpoints/CT-CLIP_v2.pt

# Set working directory to feature_extraction (entrypoint runs extract_feat_LP.sh here)
WORKDIR /opt/app/src/feature_extraction

ENTRYPOINT []
