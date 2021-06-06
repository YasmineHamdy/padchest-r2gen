# R2Gen
This is clone for https://github.com/cuhksz-nlp/R2Gen to add dataloader to padchest dataset

## Requirements

- `torch==1.8.1`
- `torchvision==0.9.1`
- `opencv-python==4.4.0.42`
- `torchtext===0.9.1`


## Download R2Gen
You can download the models we trained for each dataset from [here](https://github.com/cuhksz-nlp/R2Gen/blob/main/data/r2gen.md).

## Datasets
We use two datasets (IU X-Ray and MIMIC-CXR) in our paper.

For `IU X-Ray`, you can download the dataset from [here](https://drive.google.com/file/d/1c0BXEuDy8Cmm2jfN0YYGkQxFZd2ZIoLg/view?usp=sharing) and then put the files in `data/iu_xray`.

For `MIMIC-CXR`, you can download the dataset from [here](https://drive.google.com/file/d/1DS6NYirOXQf8qYieSVMvqNwuOlgAbM_E/view?usp=sharing) and then put the files in `data/mimic_cxr`.

For 'PadChest' you can request download from https://bimcv.cipf.es/bimcv-projects/padchest/

## Run on IU X-Ray

Run `bash run_iu_xray.sh` to train a model on the IU X-Ray data.

## Run on MIMIC-CXR

Run `bash run_mimic_cxr.sh` to train a model on the MIMIC-CXR data.

## Run on PadChest

Run `bash run_padchest.sh` to train a model on the MIMIC-CXR data.
