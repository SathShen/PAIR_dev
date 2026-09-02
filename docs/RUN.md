# Training

You can use [this script](../hf_down.py) to download Qwen2.5-VL weights to your local cache prior to training.

## Qwen backbone (`QWEN_MODEL`)

By default the model loads **Qwen2.5-VL-3B-Instruct**. You can switch to the 7B backbone (or any compatible Hugging Face Qwen2.5-VL checkpoint) via the config override:

```bash
# 3B (default)
QWEN_MODEL "Qwen/Qwen2.5-VL-3B-Instruct"

# 7B
QWEN_MODEL "Qwen/Qwen2.5-VL-7B-Instruct"
```

Pass this as an extra CLI flag on any of the training / eval commands below. Use the same `QWEN_MODEL` for training and evaluation so the ViT / LLM weights match the checkpoint.

## Cached ScanNet ViT features (`CACHE_QWEN_FEATURES`)

Our repository provides the option to load cached ViT features for ScanNet RGB images during training:

- Set `CACHE_QWEN_FEATURES True` and point `FEATURE_DIR` at the feature root.
- Features are stored as bfloat16 tensors at `FEATURE_DIR/<scene_id>/<frame>.pt` (e.g. `.../scene0051_03/140.pt`).
- On a cache hit the model skips the ViT forward; on a miss (during ScanNet training) it runs the ViT and writes the missing `.pt` files under `FEATURE_DIR`.

Precomputed ScanNet Qwen ViT features from the Qwen2.5-VL-3B will be made available on Hugging Face. After downloading them, set `FEATURE_DIR` to that directory and enable caching, for example:

```bash
CACHE_QWEN_FEATURES True \
FEATURE_DIR "/path/to/scannet_image_qwen_features" \
```

Caching applies to ScanNet datasets only. Leave `CACHE_QWEN_FEATURES False` (default) if you are are not using cached features.

We also provide a script to build the cache yourself. Run [cache_qwen_vit_features.py](cache_qwen_vit_features.py). 

```bash
# 3B ViT features
python docs/cache_qwen_vit_features.py --model 3b \
  --frames-root /path/to/SEMSEG_100k/frames_square_highres \
  --out-dir /path/to/scannet_image_qwen_features

# 7B ViT features (use a separate FEATURE_DIR from 3B)
python docs/cache_qwen_vit_features.py --model 7b \
  --frames-root /path/to/SEMSEG_100k/frames_square_highres \
  --out-dir /path/to/scannet_image_qwen_features_7b
```

To parallelize across GPUs, shard by scene (each process owns a disjoint subset of scenes):

```bash
# Example: 4 GPUs, one shard per GPU
CUDA_VISIBLE_DEVICES=0 python docs/cache_qwen_vit_features.py --model 3b \
  --frames-root /path/to/SEMSEG_100k/frames_square_highres \
  --out-dir /path/to/scannet_image_qwen_features \
  --shard-id 0 --num-shards 4 &
CUDA_VISIBLE_DEVICES=1 python docs/cache_qwen_vit_features.py --model 3b \
  --frames-root /path/to/SEMSEG_100k/frames_square_highres \
  --out-dir /path/to/scannet_image_qwen_features \
  --shard-id 1 --num-shards 4 &
# ... shards 2 and 3 similarly
wait
```

Then enable caching in training with that output directory:

```bash
CACHE_QWEN_FEATURES True \
FEATURE_DIR "/path/to/scannet_image_qwen_features" \
QWEN_MODEL "Qwen/Qwen2.5-VL-3B-Instruct" \
```

## Gradient checkpointing (`GRADIENT_CHECKPOINTING`)

Enable activation checkpointing on the Qwen backbone to reduce GPU memory (trades compute for memory):

```bash
GRADIENT_CHECKPOINTING True \
```

Default is `False`. When enabled, the model turns on non-reentrant checkpointing and `enable_input_require_grads()` for the LoRA-wrapped Qwen, and sets `use_cache=False` during training.

## 3D Grounding Only Training

```bash
source scripts/setup.sh
configure_local
BS=1 CHECKPOINT_PERIOD=8000 EVAL_PERIOD=16000 EVAL_ONLY=0 NUM_VAL_DATALOADERS=1 NUM_DATALOADERS=2 $PREFIX "${PREFIX_ARGS[@]}" scripts/main_qwen.sh \
GENERATION False \
DATASETS.TRAIN "('sr3d_ref_scannet_train_single','scanrefer_scannet_anchor_train_single','nr3d_ref_scannet_anchor_train_single',)" \
DATASETS.TEST "('sr3d_ref_scannet_val_single_batched','scanrefer_scannet_anchor_val_single_batched','nr3d_ref_scannet_anchor_val_single_batched',)" \
DATASET_MUL '[1,3,3]'
```

To test only one dataset, e.g., sr3d, add run the following command:
```bash
source scripts/setup.sh
configure_local
BS=1 CHECKPOINT_PERIOD=8000 EVAL_PERIOD=16000 EVAL_ONLY=0 NUM_VAL_DATALOADERS=1 NUM_DATALOADERS=2 $PREFIX "${PREFIX_ARGS[@]}" scripts/main_qwen.sh \
GENERATION False \
DATASETS.TRAIN "('sr3d_ref_scannet_train_single',)" \
DATASETS.TEST "('sr3d_ref_scannet_val_single_batched',)"
```

## 3D Grounding+QA Training 

```
source scripts/setup.sh
configure_local
BS=1 EVAL_PERIOD=16000 CHECKPOINT_PERIOD=4000 EVAL_ONLY=0 NUM_VAL_DATALOADERS=1 NUM_DATALOADERS=2 $PREFIX "${PREFIX_ARGS[@]}" scripts/main_qwen.sh \
GENERATION True \
DATASETS.TRAIN "('sr3d_ref_scannet_train_single','scanrefer_scannet_anchor_train_single','nr3d_ref_scannet_anchor_train_single','scanqa_ref_scannet_train_single','sqa3d_ref_scannet_train_single',)" \
DATASETS.TEST "('sr3d_ref_scannet_val_single_batched','scanrefer_scannet_anchor_val_single_batched','nr3d_ref_scannet_anchor_val_single_batched','scanqa_ref_scannet_val_single_batched','sqa3d_ref_scannet_val_single_batched')" \
DATASET_MUL '[1,3,3,1,1]'
```

## 2D-3D Training

```bash
source scripts/setup.sh
configure_local # To use SLURM: replace with "configure_slurm --partition=$SLURM_PARTITION_NAME"
BBS=1 BS2D=1 BS3D=1 CHECKPOINT_PERIOD=4000 EVAL_PERIOD=16000 EVAL_ONLY=0 NUM_VAL_DATALOADERS=1 NUM_DATALOADERS=2 $PREFIX "${PREFIX_ARGS[@]}" scripts/main_qwen.sh \
INPUT.FRAME_LEFT_2D 0 INPUT.FRAME_RIGHT_2D 0 INPUT.SAMPLING_FRAME_NUM_2D 1 \
GENERATION True \
MULTI_TASK_TRAINING True \
TRAIN_3D True \
TRAIN_2D True \
MODEL.DECODER_2D True \
MODEL.DECODER_3D True \
TEST.EVAL_3D True \
TEST.EVAL_2D True \
FORCE_DECODER_3D True \
PSEUDO_2D_AUG True \
MAX_FRAME_NUM 120 \
PROB "[0.8,0.2]" \
DATASETS.TRAIN_2D "('refcoco_train','refcoco+_train','refcocog_train','coco_2017_train','llava150k_bench','alpaca_text_bench')" \
DATASETS.TEST_2D_ONLY "('realworld_vqa_bench','coco_2017_val','refcoco_val','refcoco+_val','refcocog_val',)" \
DATASETS.TRAIN_3D "('scanqa_ref_scannet_train_single','sqa3d_ref_scannet_train_single','sr3d_ref_scannet_train_single','scanrefer_scannet_anchor_train_single','nr3d_ref_scannet_anchor_train_single','scannet200_context_instance_train_200cls_single_highres_100k','matterport_train_single',)" \
DATASETS.TEST_3D_ONLY "('matterport_val_single','locate3d_ref_scannetpp_val_single_batched','scanqa_ref_scannet_val_single_batched','sqa3d_ref_scannet_val_single_batched','sr3d_ref_scannet_val_50_single_batched','sr3d_ref_scannet_train_eval_single_batched','scanrefer_scannet_anchor_val_50_single_batched','scanrefer_scannet_anchor_train_eval_single_batched','nr3d_ref_scannet_anchor_val_50_single_batched','nr3d_ref_scannet_anchor_train_eval_single_batched','scannet200_context_instance_val_200cls_single_highres_100k',)" \
DATASETS.TRAIN "('matterport_train_single','scanqa_ref_scannet_train_single','sqa3d_ref_scannet_train_single','sr3d_ref_scannet_train_single','scanrefer_scannet_anchor_train_single','nr3d_ref_scannet_anchor_train_single','scannet200_context_instance_train_200cls_single_highres_100k','refcoco_train','refcoco+_train','refcocog_train','coco_2017_train','llava150k_bench','alpaca_text_bench')" \
DATASETS.TEST "('locate3d_ref_scannetpp_val_single_batched','scannet200_context_instance_val_200cls_single_highres_100k','scanqa_ref_scannet_val_single_batched','scanqa_ref_scannet_train_single','sqa3d_ref_scannet_val_single_batched','sqa3d_ref_scannet_train_eval_single_batched','realworld_vqa_bench','sr3d_ref_scannet_val_single_batched','scanrefer_scannet_anchor_val_single_batched','nr3d_ref_scannet_anchor_val_single_batched','refcoco_val','refcoco+_val','refcocog_val','matterport_val_single','mmlupro_text_bench',)" \
DATASET_MUL '[4,2,1,4,12,12,40,1,1,1,1,1,1]'
```

## Evaluation

To evaluate baseline on 3D grounding, replace the `$CKPT_PATH` with the path to the checkpoint.

To evaluate, simply modify the training script with `EVAL_ONLY=1`, and define the checkpoint path `$CKPT_PATH`.

For example:
```bash
CKPT_PATH="ckpts/qwen3d.pth"

# Sr3D 
BS=1 CHECKPOINT_PERIOD=8000 EVAL_PERIOD=16000 EVAL_ONLY=1 NUM_VAL_DATALOADERS=2 NUM_DATALOADERS=2 $PREFIX "${PREFIX_ARGS[@]}" scripts/main_qwen.sh \
USE_AUTO_NOUN_DETECTION False \
MODEL.WEIGHTS "$CKPT_PATH" \
DATASETS.TRAIN "('sr3d_ref_scannet_train_single',)" \
DATASETS.TEST "('sr3d_ref_scannet_val_single_batched',)" \

# ScanRefer + Nr3D
BS=1 CHECKPOINT_PERIOD=8000 EVAL_PERIOD=16000 EVAL_ONLY=1 NUM_VAL_DATALOADERS=2 NUM_DATALOADERS=2 $PREFIX "${PREFIX_ARGS[@]}" scripts/main_qwen.sh \
USE_AUTO_NOUN_DETECTION False \
MODEL.WEIGHTS "$CKPT_PATH" \
DATASETS.TRAIN "('scanrefer_scannet_anchor_train_single','nr3d_ref_scannet_anchor_train_single',)" \
DATASETS.TEST "('scanrefer_scannet_anchor_val_single_batched','nr3d_ref_scannet_anchor_val_single_batched',)" \
```

- To visualize the results, set `VISUALIZE_REF` to `True`. We use [Pyviz3D](https://github.com/francisengelmann/PyViz3D) and instructions will be printed to the console explaining how to view the results.

A complete example is below:
```bash
CKPT_PATH="ckpts/qwen3d.pth"
source scripts/setup.sh
configure_local

# Sr3D (as-is)
BS=1 EVAL_PERIOD=16000 CHECKPOINT_PERIOD=4000 EVAL_ONLY=1 NUM_VAL_DATALOADERS=1 NUM_DATALOADERS=2 $PREFIX "${PREFIX_ARGS[@]}" scripts/main_qwen.sh \
GENERATION False \
VISUALIZE_REF True \
MODEL.WEIGHTS "$CKPT_PATH" \
DATASETS.TRAIN "('sr3d_ref_scannet_train_single',)" \
DATASETS.TEST "('sr3d_ref_scannet_val_50_single_batched',)" \

# ScanRefer + Nr3D
BS=1 EVAL_PERIOD=16000 CHECKPOINT_PERIOD=4000 EVAL_ONLY=1 NUM_VAL_DATALOADERS=1 NUM_DATALOADERS=2 $PREFIX "${PREFIX_ARGS[@]}" scripts/main_qwen.sh \
USE_AUTO_NOUN_DETECTION False \
GENERATION False \
VISUALIZE_REF True \
MODEL.WEIGHTS "$CKPT_PATH" \
DATASETS.TRAIN "('scanrefer_scannet_anchor_train_single','nr3d_ref_scannet_anchor_train_single',)" \
DATASETS.TEST "('scanrefer_scannet_anchor_val_50_single_batched','nr3d_ref_scannet_anchor_val_50_single_batched',)" \
```

### Generation Evaluation
```bash
export CKPT_PATH="ckpts/ckpt.pth"
source scripts/setup.sh
configure_local
BS=1 EVAL_ONLY=1 NUM_VAL_DATALOADERS=1 NUM_DATALOADERS=2 $PREFIX "${PREFIX_ARGS[@]}" scripts/main_qwen.sh \
GENERATION True \
DATASETS.TRAIN "('scanqa_ref_scannet_train_single','sqa3d_ref_scannet_train_single',)" \
DATASETS.TEST "('scanqa_ref_scannet_val_single_batched','sqa3d_ref_scannet_val_single_batched',)"
```

## Notes

- The dataloader is CPU bound, so increase `NUM_DATALOADERS` to the number of CPUs (divided by `NUM_GPUS`) on the machine.
- Training requires a lot of CPU memory, so ensure at least 40GB per GPU. If you run out of CPU memory, try reducing `NUM_DATALOADERS` or `NUM_VAL_DATALOADERS`.
- To use SLURM, replace `configure_local` with `configure_slurm --partition=$SLURM_PARTITION_NAME`. Make sure to set the desired number of GPUs and nodes beforehand (e.g., `export NUM_GPUS=8` and `export NUM_MACHINES=2`).