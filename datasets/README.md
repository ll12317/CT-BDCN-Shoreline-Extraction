# Dataset description

The original Sentinel-2 Level-2A images used in this study were obtained from the ESA Copernicus Data Space Ecosystem.

The Vietnam shoreline labels were manually interpreted based on the instantaneous land-water boundary visible at the satellite acquisition time. Due to the large data volume and data management restrictions, the full processed Vietnam shoreline dataset is not directly uploaded to this GitHub repository. The processed image patches and manually interpreted labels are available from the corresponding author upon reasonable request.

A small dummy dataset is provided in `datasets/dummy_dataset/` to demonstrate the expected data format and running workflow.

## Expected data structure

```text
datasets/
├── train/
│   ├── images/
│   └── masks/
├── val/
│   ├── images/
│   └── masks/
└── dummy_dataset/
    ├── images/
    └── masks/

## Data availability

The full processed Vietnam shoreline dataset is not uploaded to this repository due to the large data volume and data management restrictions. The manually interpreted shoreline labels and processed sample patches are available from the corresponding author upon reasonable request.

A small dummy dataset is provided to demonstrate the required data organization and running workflow. Users can replace the dummy images and masks with their own dataset following the same folder structure.