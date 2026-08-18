import argparse
import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if os.fspath(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(PROJECT_ROOT))

from src.config import load_yaml_config


DEFAULT_BINARY_CONFIG = PROJECT_ROOT / "config" / "binary_model_v5_synaptic.yml"
DEFAULT_HEATMAP_CONFIG = PROJECT_ROOT / "config" / "heatmap_model_v5_synaptic.yml"
DEFAULT_OUTPUT = PROJECT_ROOT / "checkpoints" / "best_model_binary_heatmap_v5_synaptic.pth"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Combine binary-mask and center-heatmap model weights into one file."
    )
    parser.add_argument("--binary-config", default=os.fspath(DEFAULT_BINARY_CONFIG))
    parser.add_argument("--heatmap-config", default=os.fspath(DEFAULT_HEATMAP_CONFIG))
    parser.add_argument("--binary-checkpoint", help="Defaults to model.best_model_path in --binary-config.")
    parser.add_argument("--heatmap-checkpoint", help="Defaults to model.best_model_path in --heatmap-config.")
    parser.add_argument("--output", default=os.fspath(DEFAULT_OUTPUT))
    return parser.parse_args()


def resolve_path(path, base_dir=PROJECT_ROOT):
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return base_dir / path


def torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict) and all(torch.is_tensor(value) for value in checkpoint.values()):
        return checkpoint
    raise KeyError("Checkpoint does not contain a model_state_dict.")


def load_weights(checkpoint_path):
    checkpoint = torch_load(checkpoint_path)
    return checkpoint_state_dict(checkpoint)


def main():
    args = parse_args()

    binary_config_path = resolve_path(args.binary_config)
    heatmap_config_path = resolve_path(args.heatmap_config)
    binary_config = load_yaml_config(binary_config_path)
    heatmap_config = load_yaml_config(heatmap_config_path)

    binary_checkpoint_path = resolve_path(
        args.binary_checkpoint or binary_config["model"]["best_model_path"]
    )
    heatmap_checkpoint_path = resolve_path(
        args.heatmap_checkpoint or heatmap_config["model"]["best_model_path"]
    )
    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    combined = {
        "binary_state_dict": load_weights(binary_checkpoint_path),
        "heatmap_state_dict": load_weights(heatmap_checkpoint_path),
    }

    torch.save(combined, output_path)
    print(f"Saved combined ROI checkpoint: {output_path}")
    print(f"Binary source: {binary_checkpoint_path}")
    print(f"Heatmap source: {heatmap_checkpoint_path}")


if __name__ == "__main__":
    main()
