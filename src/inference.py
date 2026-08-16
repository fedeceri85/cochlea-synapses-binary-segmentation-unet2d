#Prediction
import torch

from monai.inferers import sliding_window_inference
from monai.transforms import Activations, AsDiscrete, Compose


import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage import measure
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed



def predict_binary(model, image, device, roi_size, sw_batch_size,
                   prediction_sigmoid, prediction_threshold):
    post_pred = Compose([
        Activations(sigmoid=prediction_sigmoid),
        AsDiscrete(threshold=prediction_threshold),
    ])

    model.eval()

    with torch.no_grad():

        output = sliding_window_inference(
            image.to(device),
            roi_size=tuple(roi_size),
            sw_batch_size=sw_batch_size,
            predictor=model,
        )

        prediction = post_pred(output)

    return prediction

from skimage.morphology import closing
def binary_to_instance_labels(binary_mask, min_size=20, min_distance=5):
    # 1. Ensure boolean mask
    mask = binary_mask > 0

   # # 2. Optional cleanup
    mask = remove_small_objects(mask, min_size=min_size)
    mask = closing(mask)  # Fill small holes in the mask
    # 3. Distance transform: high values near object centers
    distance = ndi.distance_transform_edt(mask)

    # 4. Find likely object centers
    coords = peak_local_max(
        distance,
        labels=mask,
        min_distance=min_distance,
        exclude_border=False,
    )

    # 5. Convert center points into marker image
    markers = np.zeros_like(mask, dtype=np.int32)
    markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)

    # 6. Expand markers using watershed
    labels = watershed(
        -distance,
        markers,
        mask=mask,
    )
    #Remove small objects from the labeled image
    labels = remove_small_objects(labels, min_size=min_size)

    #REset label numbers and order from left to right
    labels = measure.label(labels)
    rp = measure.regionprops(labels)
    centroids = [c.centroid[1] for c in rp]
    sorted_indices = np.argsort(centroids)
    new_labels = np.zeros_like(labels)
    for new_label, old_index in enumerate(sorted_indices, start=1):
        new_labels[labels == rp[old_index].label] = new_label

    return new_labels

def predict_labels(model, image, device, roi_size, sw_batch_size,
                   prediction_sigmoid, prediction_threshold,min_size=20, min_distance=5):

    pred = predict_binary(model, image, device, roi_size, sw_batch_size,
                          prediction_sigmoid, prediction_threshold)

    labels = binary_to_instance_labels(pred[0][0].cpu().numpy(), min_size=min_size, min_distance=min_distance)

    return labels
