Much of our data shares the same format as [UniVLG](https://github.com/facebookresearch/univlg)

# ScanNet
- For setting up the ScanNet dataset (and optionally Matterport3D dataset), follow instrictions from [here](https://github.com/ayushjain1144/odin/blob/main/data_preparation/scannet/README.md).

- Next, Download the ScanEnts3D version of the NR3D and ScanRefer dataset from [here](https://scanents3d.github.io/dataset.html) and unzip `ScanEnts3D_ScanRefer.zip`.

```bash
mkdir data/refer_it_3d
cd data/refer_it_3d
wget https://scanents3d.github.io/ScanEnts3D_Nr3D.csv https://scanents3d.github.io/ScanEnts3D_ScanRefer.zip
unzip ScanEnts3D_ScanRefer.zip
```

- Download the SR3D dataset from [here](https://drive.google.com/drive/folders/1DS4uQq7fCmbJHeE-rEbO8G1-XatGEqNV) provided by Referit3D
```bash
gdown --folder 1DS4uQq7fCmbJHeE-rEbO8G1-XatGEqNV
mv Sr3D/* .  # move all sr3d csv to the parent refer_it_3d folder
```

- Generate the train-test splits for Referi3D and ScanRefer:
```bash
export REF_DATASET='data/refer_it_3d'  # path to refer_it_3d folder
python data_preparation/refexp/make_splits.py
```

## ScanQA and SQA3D datasets
- Download the ScanQA dataset from [here](https://github.com/ATR-DBI/ScanQA?tab=readme-ov-file) and the SQA3D dataset from [here](https://github.com/SilongYong/SQA3D)

### Scannet Metadata

You can download some pre-computed data required for vision-language grounding as follows in any data directory you want to store it in:
```bash
hf download katefgroup/UniVLG --include "scannet/*" --local-dir . && mv scannet/* . && rmdir scannet
```
Then later, point `PRECOMPUTED_SCANNET_PATH` to this folder in scripts/main.sh. This is described later also.

If you want to generate it from scratch, see below.

For all scripts, set the following environment variables:
```bash
REF_DATASET="ckpts/scannet"
DATA_PATH="..."
```

### Span Prediction

To pre-compute span predictions:
```bash
uv run accelerate launch --main_process_port $RANDOM tools/generate_predicted_spans.py
```

## 2D Datasets

The above steps are enough if you just want to run inference on 3D language grounding datasets or train the 3D only baseline. However, if you want to train with 2D datasets too, you follow the below steps:

Make a new folder: `mkdir data/datasets_2d; cd data/datasets_2d`

Inside of it, download the following: 
- Download COCO dataset from [here](https://cocodataset.org/#download)
- Download the RefCOCO, RefCOCO+, and RefCOCOg datasets from [here](https://github.com/lichengunc/refer). This [issue](https://github.com/lichengunc/refer/issues/14#issuecomment-1258318183) might be relevant if download links don't work. 
- Download the 3D pointmap data for COCO dataset:

```bash
uvx --from huggingface_hub huggingface-cli download katefgroup/UniVLG_ScanNet_MonoDepth --local-dir data/datasets_2d/coco_3d_moge
```

## Dataset & checkpoint paths
Set these variables in `main.sh` to point at your local setup:
| Variable | Description |
|---|---|
| `CKPTS_PATH` | Path to checkpoints |
| `PRECOMPUTED_SCANNET_PATH` | Path to image/text embeddings and other ScanNet metadata |
| `DETECTRON2_DATASETS_2D` | Path to `datasets_2d` |
| `DETECTRON2_DATASETS` | Path to the folder containing RGB-D images from ScanNet and Matterport3D |
| `REF_DATASET` | Path to the `refer_it_3d` folder |
| `SCANNET200_DATA_DIR` | Path to ScanNet200's `train_validation_database.yaml` |
| `MATTERPORT_DATA_DIR` | Path to Matterport3D's `train_validation_database.yaml` |
| `OUTPUT_DIR_PREFIX` | Path to the folder for logs and checkpoints |

### LLaVA-Instruct-150K

The dataset mapper used for training with LLaVA-Instruct-150K data reads from the `llava_instruct_150k.json` file that can be downloaded from [here](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/tree/main)

To set the path to the json file, set the following environment variable: 

```
export LLAVA_INSTRUCT_PATH="[path-to-llava_instruct_150k.json]"
```
