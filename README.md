# CT-CLIP
This repository is based on the official repository [CT-CLIP](https://github.com/ibrahimethemhamamci/CT-CLIP) for linear probing CT-CLIP model for [CVPR 2026 CTFM challenge](https://www.codabench.org/competitions/12650/).


<p align="center">
  <img src="figures/CT-CLIP.png" width="100%">
</p>

## Building the feature-extraction image (`ctclip_lp.tar.gz`)

The image runs **fully offline**: the `Dockerfile` `COPY`s the whole repo and bakes in the
model weights, so the checkpoint must already be present at the repo root before you build.

- **Prerequisites**: Docker, and the `huggingface_hub` CLI for the downloads below
  (`pip install -U "huggingface_hub[cli]"`). The build pulls a CUDA base image + pip
  packages over the network; producing the tarball needs ~7 GB of free disk.

- **1. Download the CT-CLIP checkpoint into `checkpoints/`** — the CLIP (CTViT + text)
  weights. From the CT-RATE HuggingFace dataset (gated — accept the terms and
  `huggingface-cli login` first):
  ```bash
  huggingface-cli download ibrahimhamamci/CT-RATE \
      models/CT-CLIP-Related/CT-CLIP_v2.pt \
      --repo-type dataset --local-dir .
  mkdir -p checkpoints && mv models/CT-CLIP-Related/CT-CLIP_v2.pt checkpoints/
  ```
  Direct link: <https://huggingface.co/datasets/ibrahimhamamci/CT-RATE/blob/main/models/CT-CLIP-Related/CT-CLIP_v2.pt>
  → final path must be `checkpoints/CT-CLIP_v2.pt`.

- **2. (optional) Stage the offline text encoder into `hf_cache/`** — CT-CLIP's text branch
  loads `microsoft/BiomedVLP-CXR-BERT-specialized`. Feature extraction no longer needs it:
  `extract_feat_LP.py` builds `CTCLIP(..., image_only=True)`, which constructs neither the
  text tower nor the tokenizer, so nothing touches HuggingFace and the image builds and runs
  fine without `hf_cache/`. Stage it only if you intend to run the text branch (zero-shot,
  VocabFine) inside the container; baking it keeps that path offline too
  (`TRANSFORMERS_OFFLINE=1`):
  ```bash
  HF_HOME="$(pwd)/hf_cache" huggingface-cli download microsoft/BiomedVLP-CXR-BERT-specialized
  ```
  This populates `hf_cache/hub/models--microsoft--BiomedVLP-CXR-BERT-specialized/`.

- **3. Build the image** (`.dockerignore` keeps the context lean but intentionally keeps
  `checkpoints/` and `hf_cache/`):
  ```bash
  docker build -f Dockerfile -t ctclip_lp .
  ```

- **4. Save + compress** to produce the deliverable:
  ```bash
  docker save ctclip_lp | gzip > ctclip_lp.tar.gz
  ```

- **Load & run elsewhere** — `docker load < ctclip_lp.tar.gz`, then mount input/output
  dirs and run the extractor (checkpoint is baked at `/opt/app/checkpoints/CT-CLIP_v2.pt`):
  ```bash
  docker run --gpus all \
      -v /path/to/niftis:/workspace/inputs \
      -v /path/to/out:/workspace/outputs \
      ctclip_lp ./extract_feat_LP.sh
  ```


## Host environment with uv (matches the Docker image)

`pyproject.toml` + `uv.lock` reproduce the `Dockerfile` runtime **on the host**, so the
extraction scripts under `src/feature_extraction/` can be run without Docker and still
produce the same embeddings. Every pin is the version `pip list` reports inside the built
`ctclip_lp` image — Python 3.10.13, `torch==2.1.0+cu118`, `numpy==1.26.4`,
`transformers==4.44.2`, plus both in-repo packages installed editable, exactly as the
Dockerfile installs them.

```bash
# Creates ./.venv from uv.lock (uv fetches CPython 3.10.13 itself).
uv sync
```

Then run any extraction script through `uv run`, which puts `.venv/bin` on `PATH` so the
scripts' `python3` resolves to the pinned interpreter:

```bash
uv run src/feature_extraction/extract_feat_alldiseases_MSWAL_train.sh
```

Each script takes `DATA_ROOT` / `INPUT_DIR` / `OUTPUT_DIR` / `CHECKPOINT` / `BATCH_SIZE` /
`CUDA_VISIBLE_DEVICES` from the environment, e.g.:

```bash
INPUT_DIR=/path/to/niftis OUTPUT_DIR=/path/to/out CUDA_VISIBLE_DEVICES=0 \
    uv run src/feature_extraction/extract_feat_alldiseases_MSWAL_train.sh
```

> If you have another virtualenv active, `uv run` prints
> `VIRTUAL_ENV=... does not match the project environment path .venv and will be ignored`.
> That warning is expected and correct — the project `.venv` is what gets used.

### Two notes on how the pins were derived

- **CUDA 11.8 wheels.** The base image's torch is built against CUDA 11.8
  (`torch.version.cuda == "11.8"`), so `pyproject.toml` resolves `torch` / `torchvision`
  from the `https://download.pytorch.org/whl/cu118` index rather than PyPI (whose
  `torch==2.1.0` is a cu121 build).
- **`override-dependencies = ["numpy==1.26.4"]`.** The Dockerfile installs `numpy<2` as its
  *last* step, deliberately downgrading underneath `opencv-python==5.0.0.93`, which declares
  `numpy>=2`. pip tolerates that ordering; uv's resolver would reject it outright, so the
  override states the same end state explicitly. This reproduces the container's actual
  package set — `transformer_maskgit` does `import cv2` at import time and works fine
  against numpy 1.26.4.

### Verifying host/container parity

The uv env was checked against the built image on eight MSWAL volumes; all 512-d embeddings
came out **bit-identical**. To re-check after changing a pin:

```bash
IN=/path/to/a/few/niftis
mkdir -p /tmp/parity/host /tmp/parity/docker && chmod 777 /tmp/parity/docker

INPUT_DIR="$IN" OUTPUT_DIR=/tmp/parity/host MODEL_DIR=ctclip_lp \
    uv run src/feature_extraction/extract_feat_alldiseases_MSWAL_train.sh

docker run --rm --gpus '"device=1"' \
    -v "$IN":/workspace/inputs:ro -v /tmp/parity/docker:/workspace/outputs \
    ctclip_lp ./extract_feat_LP.sh

uv run python -c '
import h5py, numpy as np, os
h, d = "/tmp/parity/host/ctclip_lp/embeddings", "/tmp/parity/docker"
for f in sorted(os.listdir(d)):
    a = h5py.File(os.path.join(h, f))["y_hat"][:]
    b = h5py.File(os.path.join(d, f))["y_hat"][:]
    print(f, "bit-identical:", np.array_equal(a, b))
'
```

Utility packages that play no part in the forward pass do differ slightly from the
container (`requests`, `urllib3`, `certifi`, `jinja2`, `sympy`, `networkx`, `pyyaml`,
`six`, `psutil`, `pytz`, `idna`, `charset-normalizer`, `markupsafe`, `fsspec`) — they are
pulled from the conda base image inside Docker and from PyPI here. Everything the
embeddings depend on (`torch`, `torchvision`, `triton`, `numpy`, `nibabel`, `h5py`,
`einops`, `vector-quantize-pytorch`, `torchtyping`, `opencv-python`, `pillow`, `tqdm`,
`transformers`, `tokenizers`, `accelerate`, `beartype`, `ema-pytorch`, `scipy`, `pandas`)
matches exactly.


## Running feature extraction

`src/feature_extraction/extract_feat_all_test_datasets.sh` extracts every test
dataset in one go, spreading the work over both GPUs:

```bash
uv run ./src/feature_extraction/extract_feat_all_test_datasets.sh
DRYRUN=1 ./src/feature_extraction/extract_feat_all_test_datasets.sh   # show the queue
```

| env | default | meaning |
|---|---|---|
| `GPUS` | `0 1` | GPU indices to spread the queue over |
| `JOBS_PER_GPU` | `4` | concurrent units per GPU |
| `RESULTS_ROOT` | `<repo>/results` | output tree, passed to every unit |
| `LOG_DIR` | `<repo>/logs/feature_extraction/<ts>` | per-unit logs |
| `DRYRUN` | `0` | list units and exit |

**Why it is shaped this way.** The per-dataset scripts are very unequal —
AMOS_clf_tr_val alone is ~12.8k of the ~18.4k forward passes — so giving each
script a GPU would leave one idle for most of the run. The runner instead splits
the two AMOS-shaped scripts per disease (through their `DISEASES` override) into
**29 units** on a single queue that every worker pulls from, so a worker that
finishes early takes the next unit. The four non-ROI AMOS diseases stay one unit
because they share a single whole-image extraction.

`JOBS_PER_GPU` matters more than the GPU count: extraction is **CPU/IO-bound**,
not GPU-bound — nibabel's gzip read and the trilinear resample dominate, and one
unit per GPU leaves it at 2-7% utilisation. Measured on 2x RTX 4090 / 48 cores,
going from 1 to 4 units per GPU took throughput from **1.32 to 4.63 volumes/s**
(~3.5x), at ~9 GB VRAM and ~30 of 48 cores.

Units are independent: a failure is logged and the rest continue, with a summary
and a non-zero exit at the end. Note `extract_feat_LP.py` asserts each output
`.h5` does not already exist, so a unit whose embeddings are already on disk
fails rather than resuming — extract into a clean tree.

Individual datasets can still be run on their own, and all of them honour
`RESULTS_ROOT`, `INPUT_DIR`, `OUTPUT_DIR`, `CHECKPOINT`, `BATCH_SIZE` and
`CUDA_VISIBLE_DEVICES`:

```bash
uv run ./src/feature_extraction/extract_feat_alldiseases_MSWAL_train.sh
DISEASES="hydronephrosis" uv run ./src/feature_extraction/extract_feat_alldiseases.sh
```

### CT-RATE

`extract_feat_alldiseases_CTRATE.sh` covers the CT-RATE cohort at
`/media/sumin/TB7/challenges/CVPR26/CT-RATE` — 1616 whole-image volumes (no ROI
masks) producing one shared embeddings set that all 18 abnormality heads read:

```
results/CT-RATE/ctclip_lp/embeddings          <- shared by every target
results/CT-RATE/<Target_Name>/ctclip_lp/results
```

Target names come from the label CSVs and are sanitised to underscores for
directory names, matching the cluster's `dispatch_CT_CLIP_multidecays_norm.sh`
layout. `LABELS_ROOT` picks the label variant: the top-level CSVs (1616 cases)
by default, or `.../CT-RATE/small` for the 816-case set the cluster uses. A CSV
is only treated as a target if its own header carries a column of the same name,
which drops strays like `Pleural effusion copy.csv` without a hardcoded list.

Linear probing then runs through the challenge repo's dispatcher, which now
carries CT-RATE as an opt-in dataset:

```bash
DATASETS="CT-RATE" ROOT=/path/to/results \
    bash .../scripts_DECAYS2/dispatch_single_team_4_dataset.sh
```


## Requirements

For the pinned, container-matching environment use `uv sync` as described above — that is
the recommended path and the one the extraction scripts are tested against.

To install into an environment you manage yourself instead, execute the following commands:

```setup
# Navigate to the 'transformer_maskgit' directory and install the required packages
cd transformer_maskgit
pip install -e .

# Return to the root directory
cd ..

# Navigate to the 'CT_CLIP' directory and install its required packages
cd CT_CLIP
pip install -e .

# Return to the root directory
cd ..
```
Neither `setup.py` declares `torch`, so install it separately; feature extraction expects
`torch==2.1.0` (CUDA 11.8) and `numpy<2` — torch 2.1.0 is compiled against NumPy 1.x and
raises `RuntimeError: Numpy is not available` under numpy 2.x. Feature extraction
additionally needs `nibabel`, `h5py`, `tqdm` and `transformers==4.44.2` (transformers >=5.0
hard-requires torch >=2.4 and otherwise disables its PyTorch backend).

After following these steps, your environment should be properly set up with all required packages.

The CT-CLIP model necessitates the use of an A100 GPU with 80GB of VRAM for a batch size of 8 for efficient training, due to the model's considerable size. Inference can be done in smaller GPUs. The patch sizes of the image encoder can be adjusted to make it fit onto smaller GPUs, although this will affect the model performance in smaller pathologies. Batch size can also be lowered, but this is not recommended for CLIP training as it will not learn negative images with lower batch sizes.

## Training

For details on the training of zero-shot CT-CLIP and fine-tuned CT-CLIP models, please navigate to [scripts](scripts).

For details on the training of text classifier, please navigate to [text_classifier](text_classifier).

## Inference

For details on the inference and evaluation of zero-shot CT-CLIP and fine-tuned CT-CLIP models, please navigate to [scripts](scripts).

For details on the inference of text classifier, please navigate to [text_classifier](text_classifier).

Inference with CT-CLIP (zero-shot) and CT-CLIP (VocabFine) takes approximately 1.5 seconds to assess 18 pathologies from a single CT volume, while inference with CT-CLIP (ClassFine) takes just 0.5 seconds for the same task.


## Pretrained Models

For your convenience, we provide access to pretrained models directly. These models have been trained on our paired radiological report and chest CT volume dataset, as elaborated in the paper.

You can download the models from the following links:

- **CT-CLIP**: [Download Here](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE/blob/main/models/CT-CLIP-Related/CT-CLIP_v2.pt)

- **CT-CLIP (VocabFine)**: [Download Here](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE/blob/main/models/CT-CLIP-Related/CT_VocabFine_v2.pt)

- **CT-CLIP (ClassFine)**: [Download Here](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE/blob/main/models/CT-CLIP-Related/CT_LiPro_v2.pt)
  
- **Text Classifier Model**: [Download Here](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE/blob/main/models/RadBertClassifier.pth)

By leveraging these pretrained models, you can easily reproduce our results or further extend our work.


## Our Dataset (CT-RATE)

A major challenge in computational research in 3D medical imaging is the lack of comprehensive datasets. Addressing this issue, we present CT-RATE, the first 3D medical imaging dataset that pairs images with textual reports. CT-RATE consists of 25,692 non-contrast chest CT volumes, expanded to 50,188 through various reconstructions, from 21,304 unique patients, along with corresponding radiology text reports, multi-abnormality labels, and metadata. We divided the cohort into two groups: 20,000 patients were allocated to the training set and 1,304 to the validation set. Our folders are structured as split_patientID_scanID_reconstructionID. For instance, "valid_53_a_1" indicates that this is a CT volume from the validation set, scan "a" from patient 53, and reconstruction 1 of scan "a". This naming convention applies to all files.

<p align="center">
  <img src="figures/CT-RATE.png" width="100%">
</p>

You can download the dataset used in this work via the [Hugging Face repository](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE). 

Data used to finetune and validate the text classifier model can be accessed [here](text_classifier/data).


## Citing Us
If you use CT-RATE or CT-CLIP, we would appreciate your references to [our paper](https://arxiv.org/abs/2403.17834).


## License
We are committed to fostering innovation and collaboration in the research community. To this end, all elements of CT-CLIP are released under a [Creative Commons Attribution (CC-BY-NC-SA) license](https://creativecommons.org/licenses/by-nc-sa/4.0/). This licensing framework ensures that our contributions can be freely used for non-commercial research purposes, while also encouraging contributions and modifications, provided that the original work is properly cited and any derivative works are shared under similar terms.
