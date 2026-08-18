import argparse
import os
import tempfile
from pathlib import Path

_MPLCONFIGDIR = Path(tempfile.gettempdir()) / "matplotlib"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", os.fspath(_MPLCONFIGDIR))

import numpy as np
import tifffile
import torch
import yaml
from monai.inferers import sliding_window_inference
from monai.networks.nets import AttentionUnet, UNet
from monai.transforms import Activations, AsDiscrete, Compose
from scipy import ndimage as ndi
from skimage import measure
from skimage.feature import peak_local_max
from skimage.morphology import closing, remove_small_objects
from skimage.segmentation import watershed


HERE = Path(__file__).resolve().parent
DEFAULT_MODEL_NAME = "checkpoints/best_model_binary_heatmap_v5_synaptic.pth"
DEFAULT_BINARY_CONFIG = "config/binary_model_v5_synaptic.yml"
DEFAULT_HEATMAP_CONFIG = "config/heatmap_model_v5_synaptic.yml"
DEFAULT_POSTPROCESSING = {
    "min_size": 20,
    "min_distance": 5,
    "heatmap_threshold": 0.3,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run two-stage cochlear synapse ROI inference from one 2D TIFF image."
    )
    parser.add_argument("image", help="Path to a single 2D TIFF image.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help="Combined ROI checkpoint. Defaults to best_model_binary_heatmap_v5_synaptic.pth.",
    )
    parser.add_argument(
        "--binary-config",
        default=DEFAULT_BINARY_CONFIG,
        help="Binary model YAML config.",
    )
    parser.add_argument(
        "--heatmap-config",
        default=DEFAULT_HEATMAP_CONFIG,
        help="Heatmap model YAML config.",
    )
    parser.add_argument("--output", help="Output ROI label TIFF path.")
    parser.add_argument(
        "--save-intermediates",
        action="store_true",
        help="Also save binary mask, heatmap prediction, and marker image.",
    )
    parser.add_argument("--binary-output", help="Binary mask output path.")
    parser.add_argument("--heatmap-output", help="Heatmap output path.")
    parser.add_argument("--markers-output", help="Marker image output path.")
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default=None,
        help="Inference device. Defaults to training.device in the binary config.",
    )
    parser.add_argument("--roi-size", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--sw-batch-size", type=int)
    parser.add_argument("--binary-threshold", type=float)
    parser.add_argument("--heatmap-threshold", type=float)
    parser.add_argument("--min-size", type=int)
    parser.add_argument("--min-distance", type=int)
    return parser.parse_args()


def resolve_existing_path(path):
    path = Path(path).expanduser()
    if path.is_absolute():
        return path

    candidates = [
        HERE / path,
        HERE  / path.name,
        Path.cwd() / path,
        Path.cwd()  / path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_yaml_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def select_device(device_name):
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if device_name == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is not available.")
    return torch.device(device_name)


def torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def build_model(model_config):
    model_kwargs = {
        "spatial_dims": model_config["spatial_dims"],
        "in_channels": model_config["in_channels"],
        "out_channels": model_config["out_channels"],
        "channels": tuple(model_config["channels"]),
        "strides": tuple(model_config["strides"]),
    }

    architecture = model_config["architecture"]
    if architecture == "unet":
        return UNet(**model_kwargs, num_res_units=model_config["num_res_units"])
    if architecture == "attention_unet":
        return AttentionUnet(**model_kwargs)
    raise ValueError(f"Unknown model architecture: {architecture}")


def branch_state_dict(checkpoint, branch_name):
    weights_key = f"{branch_name}_state_dict"
    if weights_key in checkpoint:
        return checkpoint[weights_key]
    if branch_name in checkpoint and "state_dict" in checkpoint[branch_name]:
        return checkpoint[branch_name]["state_dict"]
    raise KeyError(f"Combined checkpoint does not contain {weights_key}.")


def branch_model_config(checkpoint, branch_name):
    if branch_name in checkpoint and "model_config" in checkpoint[branch_name]:
        return checkpoint[branch_name]["model_config"]
    raise KeyError(f"Missing model config for {branch_name}.")


def branch_validation_config(checkpoint, branch_name):
    if branch_name in checkpoint and "validation_config" in checkpoint[branch_name]:
        return checkpoint[branch_name]["validation_config"]
    raise KeyError(f"Missing validation config for {branch_name}.")


def load_branch(checkpoint, branch_name, model_config, device):
    model = build_model(model_config).to(device)
    model.load_state_dict(branch_state_dict(checkpoint, branch_name))
    model.eval()
    return model


def load_image(image_path):
    image = tifffile.imread(image_path).squeeze().astype(np.float32)
    if image.ndim != 2:
        raise ValueError(f"Expected one 2D image, got shape {image.shape}.")
    if image.max() > image.min():
        image = (image - image.min()) / (image.max() - image.min())
    return image


def predict_binary(model, image, validation_config, device, roi_size=None, sw_batch_size=None, threshold=None):
    image_tensor = torch.from_numpy(image)[None, None].to(device)
    threshold = validation_config["prediction_threshold"] if threshold is None else threshold
    post_pred = Compose([
        Activations(sigmoid=validation_config["prediction_sigmoid"]),
        AsDiscrete(threshold=threshold),
    ])

    with torch.no_grad():
        output = sliding_window_inference(
            image_tensor,
            roi_size=tuple(roi_size or validation_config["roi_size"]),
            sw_batch_size=sw_batch_size or validation_config["sw_batch_size"],
            predictor=model,
        )
        prediction = post_pred(output)

    return prediction[0, 0].cpu().numpy().astype(np.uint8)


def predict_heatmap(model, image, validation_config, device, roi_size=None, sw_batch_size=None):
    image_tensor = torch.from_numpy(image)[None, None].to(device)
    post_pred = Compose([
        Activations(sigmoid=validation_config["prediction_sigmoid"]),
    ])

    with torch.no_grad():
        output = sliding_window_inference(
            image_tensor,
            roi_size=tuple(roi_size or validation_config["roi_size"]),
            sw_batch_size=sw_batch_size or validation_config["sw_batch_size"],
            predictor=model,
            overlap=0.5,
            mode="gaussian",
        )
        prediction = post_pred(output)

    return prediction[0, 0].cpu().numpy().astype(np.float32)


def heatmap_to_marker_image(heatmap, min_distance=5, threshold_abs=0.3):
    centers = peak_local_max(
        heatmap,
        min_distance=min_distance,
        threshold_abs=threshold_abs,
    )
    markers = np.zeros_like(heatmap, dtype=np.int32)
    for marker_id, (y, x) in enumerate(centers, start=1):
        markers[y, x] = marker_id
    return markers


def remove_small_components(labels, min_size):
    if min_size <= 0:
        return labels
    try:
        return remove_small_objects(labels, max_size=min_size - 1)
    except TypeError:
        return remove_small_objects(labels, min_size=min_size)


def binary_to_instance_labels(binary_mask, min_size=20, min_distance=5, markers=None):
    mask = binary_mask > 0
    mask = remove_small_components(mask, min_size=min_size)
    mask = closing(mask)
    distance = ndi.distance_transform_edt(mask)

    if markers is None:
        coords = peak_local_max(
            distance,
            labels=mask,
            min_distance=min_distance,
            exclude_border=False,
        )
        markers = np.zeros_like(mask, dtype=np.int32)
        markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)

    labels = watershed(-distance, markers, mask=mask)
    labels = remove_small_components(labels, min_size=min_size)
    labels = measure.label(labels)

    props = measure.regionprops(labels)
    sorted_props = sorted(props, key=lambda prop: prop.centroid[1])
    relabeled = np.zeros_like(labels, dtype=np.uint32)
    for new_label, prop in enumerate(sorted_props, start=1):
        relabeled[labels == prop.label] = new_label
    return relabeled


def default_output_path(image_path, suffix):
    image_path = Path(image_path)
    if image_path.suffix.lower() in (".tif", ".tiff"):
        return image_path.with_name(f"{image_path.stem}_{suffix}{image_path.suffix}")
    return image_path.with_name(f"{image_path.name}_{suffix}.tif")


def save_outputs(args, binary_mask, heatmap, markers, roi_labels):
    roi_output = Path(args.output) if args.output else default_output_path(args.image, "rois")
    tifffile.imwrite(roi_output, roi_labels.astype(np.uint32))
    outputs = [roi_output]

    if args.save_intermediates or args.binary_output:
        output = Path(args.binary_output) if args.binary_output else default_output_path(args.image, "binary")
        tifffile.imwrite(output, binary_mask.astype(np.uint8))
        outputs.append(output)

    if args.save_intermediates or args.heatmap_output:
        output = Path(args.heatmap_output) if args.heatmap_output else default_output_path(args.image, "heatmap")
        tifffile.imwrite(output, heatmap.astype(np.float32))
        outputs.append(output)

    if args.save_intermediates or args.markers_output:
        output = Path(args.markers_output) if args.markers_output else default_output_path(args.image, "markers")
        tifffile.imwrite(output, markers.astype(np.uint32))
        outputs.append(output)

    return outputs


def main():
    args = parse_args()
    model_path = resolve_existing_path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Combined model checkpoint not found: {model_path}")

    binary_config_path = resolve_existing_path(args.binary_config)
    heatmap_config_path = resolve_existing_path(args.heatmap_config)
    if not binary_config_path.exists():
        raise FileNotFoundError(f"Binary config not found: {binary_config_path}")
    if not heatmap_config_path.exists():
        raise FileNotFoundError(f"Heatmap config not found: {heatmap_config_path}")

    binary_config = load_yaml_config(binary_config_path)
    heatmap_config = load_yaml_config(heatmap_config_path)

    configured_device = args.device or binary_config.get("training", {}).get("device", "auto")
    device = select_device(configured_device)
    checkpoint = torch_load(model_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError("Combined model checkpoint must be a dictionary of state dicts.")

    binary_model_config = binary_config["model"]
    heatmap_model_config = heatmap_config["model"]
    if "binary" in checkpoint and "model_config" in checkpoint["binary"]:
        binary_model_config = branch_model_config(checkpoint, "binary")
    if "heatmap" in checkpoint and "model_config" in checkpoint["heatmap"]:
        heatmap_model_config = branch_model_config(checkpoint, "heatmap")

    binary_validation_config = binary_config["validation"]
    heatmap_validation_config = heatmap_config["validation"]
    if "binary" in checkpoint and "validation_config" in checkpoint["binary"]:
        binary_validation_config = branch_validation_config(checkpoint, "binary")
    if "heatmap" in checkpoint and "validation_config" in checkpoint["heatmap"]:
        heatmap_validation_config = branch_validation_config(checkpoint, "heatmap")

    binary_model = load_branch(checkpoint, "binary", binary_model_config, device)
    heatmap_model = load_branch(checkpoint, "heatmap", heatmap_model_config, device)
    image = load_image(args.image)

    postprocessing = dict(DEFAULT_POSTPROCESSING)
    postprocessing.update(binary_config.get("postprocessing", {}))
    postprocessing.update(heatmap_config.get("postprocessing", {}))
    postprocessing.update(checkpoint.get("postprocessing", {}))
    min_size = args.min_size if args.min_size is not None else postprocessing.get("min_size", 20)
    min_distance = (
        args.min_distance if args.min_distance is not None else postprocessing.get("min_distance", 5)
    )
    heatmap_threshold = (
        args.heatmap_threshold
        if args.heatmap_threshold is not None
        else postprocessing.get("heatmap_threshold", 0.3)
    )

    binary_mask = predict_binary(
        binary_model,
        image,
        binary_validation_config,
        device,
        roi_size=args.roi_size,
        sw_batch_size=args.sw_batch_size,
        threshold=args.binary_threshold,
    )
    heatmap = predict_heatmap(
        heatmap_model,
        image,
        heatmap_validation_config,
        device,
        roi_size=args.roi_size,
        sw_batch_size=args.sw_batch_size,
    )
    markers = heatmap_to_marker_image(
        heatmap,
        min_distance=min_distance,
        threshold_abs=heatmap_threshold,
    )
    roi_labels = binary_to_instance_labels(
        binary_mask,
        min_size=min_size,
        min_distance=min_distance,
        markers=markers,
    )

    outputs = save_outputs(args, binary_mask, heatmap, markers, roi_labels)
    print(f"Device: {device}")
    print(f"Loaded model: {model_path}")
    print(f"Binary config: {binary_config_path}")
    print(f"Heatmap config: {heatmap_config_path}")
    print(f"ROIs: {int(roi_labels.max())}")
    for output in outputs:
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
