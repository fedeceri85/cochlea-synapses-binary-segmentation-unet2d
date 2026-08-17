#Validation code

from monai.inferers import sliding_window_inference
import torch
from monai.metrics import DiceMetric
from monai.transforms import Activations, AsDiscrete, Compose


def validate_binary(model, loader, metric, device,
                    prediction_sigmoid, prediction_threshold,
                    label_threshold, roi_size, sw_batch_size):
    post_pred = Compose([
        Activations(sigmoid=prediction_sigmoid),
     #   AsDiscrete(threshold=prediction_threshold),
    ])
    post_label = Compose([
      #  AsDiscrete(threshold=label_threshold),
    ])

    model.eval()
    metric.reset()

    with torch.no_grad():
        for batch in loader:
            image = batch['image'].to(device)
            label = batch['label'].to(device)

            output = sliding_window_inference(
                image,
                roi_size=tuple(roi_size),
                sw_batch_size=sw_batch_size,
                predictor=model
            )

            pred = post_pred(output)
            gt = post_label(label)

            metric(pred,gt)
    score =metric.aggregate().item()
    metric.reset()

    return score
