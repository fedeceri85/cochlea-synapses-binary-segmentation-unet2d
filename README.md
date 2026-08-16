# Cochlea Hair Cell Binary Segmentation U-Net 2D

[![Hugging Face Model](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-FFD21E)](https://huggingface.co/fedeceri/cochlea-hair-cell-binary-segmentation-UNET2d)


Small MONAI/PyTorch pipeline for 2D binary segmentation of cochlear microscopy TIFF images, with optional watershed post-processing to produce instance labels.

Input: single-channel `*_Avg.tif` image.  
Output: binary foreground mask, or instance-label mask after watershed post-processing.



## Files

```text
config/                       # training configs
checkpoints/                  # local checkpoints
scripts/train_binary.py       # config-driven training
huggingface/infer_binary.py   # single-image inference demo for HF release
src/                          # data, model, transforms, train/validate helpers
binary_check_predictions.ipynb
huggingface/README.md         # short model card for HF upload
```

## Train

Training requires a config file. Epochs, workers, checkpoint paths, model settings, augmentation flags/probabilities, split settings, and data paths are read from YAML.

```bash
cd MONAI-test
../.venv/bin/python scripts/train_binary.py config/binary_model_v2.yml
```

Training starts from scratch when `training.resume: false`. When `training.resume: true`, it continues from the checkpoint epoch and stops at `training.epochs`.

## Inference

The inference demo is designed for the Hugging Face release folder, where the script sits next to `config.yml` and the packaged checkpoint named by `model.best_model_path`.

```bash
python infer_binary.py input_Avg.tif --output prediction_mask.tif
```

If `--output` is omitted, the script writes `<input>_prediction.tif`.

To write watershed-separated instance labels:

```bash
python infer_binary.py input_Avg.tif --output-type labels --output prediction_labels.tif
```

## Hugging Face

Only upload the selected best model to Hugging Face:

```text
README.md
infer_binary.py
best_model_binary_v1.pth
config.yml
```

Keep training code on GitHub. Copy `huggingface/README.md` to the Hugging Face repo as `README.md`, then update the GitHub URL, commit SHA, chosen checkpoint, and validation Dice.
