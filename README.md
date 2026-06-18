# CT-CLIP
This repository is based on the official repository [CT-CLIP](https://github.com/ibrahimethemhamamci/CT-CLIP) for linear probing CT-CLIP model for [CVPR 2026 CTFM challenge](https://www.codabench.org/competitions/12650/).


<p align="center">
  <img src="figures/CT-CLIP.png" width="100%">
</p>

## Building the feature-extraction image (`ctclip_lp.tar.gz`)

**fully offline**. The
`Dockerfile` `COPY`s the whole repo and bakes in the model weights, so both the
checkpoint and the text encoder must already be present at the repo root before you build.

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

- **2. Stage the offline text encoder into `hf_cache/`** — CT-CLIP's text branch loads
  `microsoft/BiomedVLP-CXR-BERT-specialized`; baking it lets the container skip all
  network access at run time (`TRANSFORMERS_OFFLINE=1`):
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


## Requirements

Before you start, you must install the necessary dependencies. To do so, execute the following commands:

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
