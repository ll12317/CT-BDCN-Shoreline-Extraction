# CT-BDCN-Shoreline-Extraction

This repository provides the PyTorch implementation of CT-BDCN, a boundary-enhanced CNN-Transformer network for shoreline extraction from Sentinel-2 remote sensing imagery.

The proposed method introduces a U-Net-BDCN-style multi-scale boundary branch, gated boundary feature enhancement, and boundary residual refinement to improve shoreline localization in complex coastal environments.

## Repository structure

```text
CT-BDCN-Shoreline-Extraction/
├── data.py
├── models/
├── scripts/
│   ├── train_main_models/
│   ├── val_main_models/
│   ├── train_ablation/
│   └── val_ablation/
├── utils/
├── datasets/
├── examples/
├── results/
├── checkpoints/
├── requirements.txt
├── environment.yml
├── LICENSE
└── README.md
Implemented models

This repository includes the training and validation code for the following main models:

CT-BDCN
CT
U-Net
DeepLabV3+
SegFormer

The repository also includes training and validation code for the following ablation variants:

CT
CT-UNet
CT-DLV3+
CT-BDCN
CT-BDCN-CBAM
Requirements

The code was developed and tested under the following environment:

Operating system: Windows 11
Programming language: Python 3.11.0
Deep learning framework: PyTorch 2.4.0
GPU: NVIDIA GeForce RTX 3090, 24 GB VRAM
CUDA: 12.8

Install the required packages using:
pip install -r requirements.txt


Data organization
The dataset should be organized as follows:
datasets/
├── train/
│   ├── images/
│   └── masks/
├── val/
│   ├── images/
│   └── masks/
Each image should have a corresponding binary mask with the same file name.

A dummy dataset is provided in datasets/dummy_dataset/ to demonstrate the expected data format and running workflow.

Training main models

To train the proposed CT-BDCN model, run:
python scripts/train_main_models/train_ct_bdcn.py
To train the other comparison models, run:
python scripts/train_main_models/train_ct.py
python scripts/train_main_models/train_unet.py
python scripts/train_main_models/train_deeplabv3p+.py
python scripts/train_main_models/train_segformer.py
Validating main models

To validate the proposed CT-BDCN model, run:
python scripts/val_main_models/val_ct_bdcn.py
To validate the other comparison models, run:
python scripts/val_main_models/val_ct.py
python scripts/val_main_models/val_unet.py
python scripts/val_main_models/val_deeplabv3+.py
python scripts/val_main_models/val_segformer.py
Training ablation models

To reproduce the ablation experiments, run:
python scripts/train_ablation/train_ct.py
python scripts/train_ablation/train_ct_unet.py
python scripts/train_ablation/train_ct_dlv3+.py
python scripts/train_ablation/train_ct_bdcn.py
python scripts/train_ablation/train_ct_bdcn_cbam.py
Validating ablation models

To validate the ablation models, run:
python scripts/val_ablation/val_ct.py
python scripts/val_ablation/val_ct_unet.py
python scripts/val_ablation/val_ct_dlv3+.py
python scripts/val_ablation/val_ct_bdcn.py
python scripts/val_ablation/val_ct_bdcn_cbam.py
Evaluation metrics

The validation scripts calculate the following metrics:

PA
Precision
Recall
F1 score
IoU
IoU_edge

The IoU_edge metric is implemented using a morphology-based boundary-band evaluation strategy.

Notes on testing

This repository provides training and validation scripts rather than independent test scripts. The experiments in the paper were conducted using the training and validation split described in the manuscript.

Data availability

The Sentinel-2 Level-2A imagery used in this study was obtained from the ESA Copernicus Data Space Ecosystem.

The manually interpreted Vietnam shoreline labels and processed sample patches are available from the corresponding author upon reasonable request due to data management restrictions.

The public SNOWED dataset used for generalization evaluation is available from its original publication.

Results

The main quantitative results are summarized in the results/ folder.

License

This project is released under the MIT License.
