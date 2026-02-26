import os
from glob import glob
import joblib
import argparse

import torch
import torch.nn as nn
import torchvision.models as models

from fogvision import gpuutils
from fogvision import fogimageclass

import importlib
from importlib import resources
import pandas as pd
import os

from pathlib import Path

from sklearn.metrics import confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def load_head(filename, device):
    pkg = "fogvision.models"
    with resources.files(pkg).joinpath(filename).open("rb") as f:
        if "cpu" in str(device):
            return torch.load(f, map_location=device, weights_only=False)
        else:
            return joblib.load(f)

def main():
    parser = argparse.ArgumentParser(description="Run fogvision on a folder of images.")
    parser.add_argument("image_folder", help="Folder containing images (e.g. data/images)")
    parser.add_argument("--save-csv-to", default=None, help="Folder containing resulting .csv")
    parser.add_argument("--sitename", default=None, help="The name of the site")
    parser.add_argument("--crop-size", type=int, default=None,
                        help="Square crop size; default is largest 32-divisible crop.")
    parser.add_argument("--random-crop", action="store_true", default=False,
                        help="Use random crop instead of center crop.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Decision threshold on fog probability (if you want to override).")
    args = parser.parse_args()

    # initialize embedder 
    basemodel = models.resnet50(pretrained=True)

    num_ftrs = basemodel.fc.in_features
    basemodel.fc = nn.Identity()  # Set the classification layer to identity to output embeddings

    # choose device
    device = gpuutils.get_gpu_most_memory() # if gpus avaible get one with most memory
    print(f"Device: {device}")

    # put the model in evaluation mode
    model = basemodel.to(device)
    model.eval()

    # then:
    if 'cpu' in str(device):
        diurnal_classif_head = load_head("fogvision_inference_head_model_diurnal_cpu.pkl", device)
        nocturnal_classif_head = load_head("fogvision_inference_head_model_nocturnal_cpu.pkl", device)
    else:
        diurnal_classif_head = load_head("fogvision_inference_head_model_diurnal.pkl", device)
        nocturnal_classif_head = load_head("fogvision_inference_head_model_nocturnal.pkl", device)


    importlib.reload(fogimageclass)
    imgpath = Path(args.image_folder)

    out_dir = imgpath

    out_dir = Path(args.save_csv_to) if args.save_csv_to else imgpath

    out_csv = out_dir / "classification_results.csv"

    print(f"Starting predictions on site {imgpath}")

    image_fns = glob(os.path.join(f"{imgpath}", "**", '*.jpg'), recursive=True)

    crop_size = args.crop_size # if crop size is None then it will take largest crop possible
    random_crop = args.random_crop
    threshold = args.threshold

    results = []

    for fn in image_fns:
        fog_img_class = fogimageclass.FogImage(filepath=fn, crop_size=crop_size)
        if fog_img_class.nocturnal:
            fog_img_class.get_fog_val(model=nocturnal_classif_head, embedding_model=basemodel, random_crop=random_crop, threshold=threshold) # NOTE only takes image embedding from 
            # fog_img_class.get_fog_val_multiple_regions(model=nocturnal_classif_head, embedding_model=basemodel) # NOTE not ideal for images with a lot of open sky works best on forested environments
        else:
            fog_img_class.get_fog_val(model=diurnal_classif_head, embedding_model=basemodel, random_crop=random_crop, threshold=threshold)
            # fog_img_class.get_fog_val_multiple_regions(model=diurnal_classif_head, embedding_model=basemodel)

        # fog_img_class.plot_image(plot_crop=False) # plot image with nocturnal and fog value

        basename = os.path.basename(fn) # filename
        path_parts = fn.split(os.sep)
        provided = args.sitename

        sitename = ( # sitename
            provided if provided else # if sitename arg is provided
            path_parts[-4] if len(path_parts) >= 4 else # if the filepath is fewer than 4 segments
            'unknown' # else unknown
        )
        
        timestamp_str = fog_img_class.timestamp.strftime('%Y-%m-%d %H:%M:%S') if fog_img_class.timestamp else 'unknown'
        
        # extract fog probability (index 1 is fog class probability)
        fog_proba = float(fog_img_class.probabilities[1].cpu().item()) if hasattr(fog_img_class.probabilities, 'cpu') else float(fog_img_class.probabilities[1])

        # extract parent folder name and determine true label
        parent_folder = os.path.basename(os.path.dirname(fn)).lower()
        if 'clear' in parent_folder:
            true_label = 0
        elif 'fog' in parent_folder:
            true_label = 1
        else:
            true_label = 'NA'  # Unknown/unlabeled
        
        # store results
        results.append({
            'timestamp': timestamp_str,
            'fn': basename,
            'sitename': sitename,
            'fog_val': fog_img_class.fog_val,
            'fog_proba': fog_proba,
            'true_label': true_label,
        })

        df = pd.DataFrame(results)
        df.to_csv(f"{imgpath}classification_results.csv", index=False)
        print(f"Image {len(df)} on Site {sitename} - fog_proba: {fog_proba}")

    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)
    print(f"Results saved to {out_csv} with {len(results)} records")