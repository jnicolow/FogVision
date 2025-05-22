"""
This module contains code to create and image class

Author: Joel Nicolow, Information and Computer Science, University of Hawaii at Manoa (September 24, 2024)
"""
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches # for plotting box of crop
from datetime import datetime


import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import functional as F
# import torch.nn.functional as F

from sklearn.base import BaseEstimator # most sklearn models inherit from this class

from fogvision.imageclass import CamImage
from fogvision import gpuutils



class TopCrop:
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size  # (height, width)

    def __call__(self, img):
        # Image dimensions
        width, height = F.get_image_size(img)
        crop_height, crop_width = self.size

        if crop_height > height or crop_width > width:
            raise ValueError("Requested crop size exceeds image dimensions.")

        # Top crop (take from the top of the image)
        top = 0
        left = (width - crop_width) // 2  # Center horizontally
        return F.crop(img, top, left, crop_height, crop_width)


class TopLeftCornerCrop:
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size  # (height, width)

    def __call__(self, img):
        width, height = F.get_image_size(img)
        crop_height, crop_width = self.size

        if crop_height > height or crop_width > width:
            raise ValueError("Requested crop size exceeds image dimensions.")

        top = 0
        left = 0  # Align to left edge
        return F.crop(img, top, left, crop_height, crop_width)
    

class FogImage(CamImage):
    def __init__(self, filepath, nocturnal=None, fog_val=None, crop_size=None, device=None):
        super().__init__(filepath)
        self.get_timestamp() # this will set self.timestamp and also returns it
        self.nocturnal = nocturnal # if 1 that means its a nocturnal image if 0 its not (meaning its diurnal)
        if self.nocturnal is None: 
            if 'diurnal' in filepath: self.nocturnal = 0 # if there is diunral in the path then it is daytime which is class zero
            elif 'nocturnal' in filepath: self.nocturnal = 1
            else:
                self.get_is_nocturnal()
        
        self.fog_val = fog_val

        if self.fog_val is None: 
            folder = os.path.basename(os.path.dirname(filepath))
            if 'clearOfFog'.lower() == folder.lower(): self.fog_val = 0 # if the image is is a training folder for a class that means the fog val is known and can be used. Othersiwse it needs to be aquired with a model prediction
            elif 'fog'.lower() == folder.lower(): self.fog_val = 1

        self.embedding = None # only used if embedding is need for multiple classifications (fog val and nocturnal or diurnal for example)

        if device is None: self.device = gpuutils.get_gpu_most_memory()
        else: self.device = device

        self.crop_size = crop_size # if none this means it will take the largest that it can


    def __repr__(self):
        return f"ExtendedCamImage(filepath={self.filepath}, timestamp={self.timestamp}, DoN={self.nocturnal}, fog_val={self.fog_val})"


    def get_timestamp(self, round_timestamp=True):
        # return super().get_timestamp() # this uses the supporting function get timestamp but we want to do based on fn
        self.timestamp = None

        filename = os.path.basename(self.filepath)
        match1 = re.search(r'(\d{4}-\d{2}-\d{2})_(\d{6})', filename)
        if match1:
            date_str = match1.group(1)
            time_str = match1.group(2)
            # Convert to datetime
            self.timestamp = datetime.strptime(date_str + time_str, '%Y-%m-%d%H%M%S')
        
        # Check for the second format: honouliuli_2022_01_01_070025.jpg
        match2 = re.search(r'(\d{4})_(\d{2})_(\d{2})_(\d{6})', filename)
        if match2:
            date_str = f"{match2.group(1)}-{match2.group(2)}-{match2.group(3)}"
            time_str = match2.group(4)
            # Convert to datetime
            self.timestamp = datetime.strptime(date_str + time_str, '%Y-%m-%d%H%M%S')
    
        
        if not self.timestamp is None and round_timestamp:
            self.round_timestamp_to_nearest_minute() # round to nearest minute


        return self.timestamp


    def round_timestamp_to_nearest_minute(self):
        if self.timestamp is not None:
            # round down to the nearest minute by setting seconds and microseconds to zero
            self.timestamp = self.timestamp.replace(second=0, microsecond=0)


    def get_is_nocturnal(self):
        # could use a model if neccesay
        # model.eval()
        # with torch.no_grad():
        #     logits, _ = model(self.to_tensor()) # the second output is the embedding
        #     probabilities = torch.sigmoid(logits)
        
        #     # Convert probabilities to binary prediction (0 or 1)
        #     self.nocturnal = int((probabilities > 0.5).float())

        # if image is gray scale its a nocturnal image
        image_np = self.to_numpy()
        height, width, _ = image_np.shape
        center_x, center_y = width // 2, height // 2
        half_size = 50
        center_region = image_np[center_y-half_size:center_y+half_size, center_x-half_size:center_x+half_size]
        # check if center region is gray
        if np.all(center_region[:, :, 0] == center_region[:, :, 1]) and np.all(center_region[:, :, 1] == center_region[:, :, 2]):
            # image is grascale
            self.nocturnal = 1
        else:
            # is not a grascale image
            self.nocturnal = 0
    
        return self.nocturnal
    

    def _get_padding(self, crop_size):
        width, height = self.size
        pad_left = max((crop_size - width) // 2, 0)
        pad_top = max((crop_size - height) // 2, 0)
        pad_right = max(crop_size - width - pad_left, 0)
        pad_bottom = max(crop_size - height - pad_top, 0)
        return (pad_left, pad_top, pad_right, pad_bottom)
    

    def to_tensor(self, random_crop=False, crop_size=None, unsqueeze:bool=False):
        if not crop_size is None: self.crop_size = crop_size # the crop_size can be defined at this stage to 
        if self.crop_size is None and crop_size is None:
            # find largest square in image
            min_size = min(self.image_size())
            self.crop_size = int(np.floor(min_size / 32)) * 32 # take largest square with size divisable by 32
            # find 32 devisable of it

        # Padding to ensure image is at least crop_size in both dimensions
        padding_transform = transforms.Lambda(
            lambda img: transforms.functional.pad(
                img,
                padding=self._get_padding(self.crop_size),
                fill=0,
                padding_mode='constant'
            )
        )

        crop = TopLeftCornerCrop(self.crop_size)

        if random_crop:
            crop = transforms.RandomCrop(self.crop_size)
        else:
            crop = transforms.CenterCrop(self.crop_size)       # crop to crop_sizexcrop_size (ResNet input size) crops the center of the image
            # crop = TopCrop(self.crop_size)


        preprocess = transforms.Compose([
            padding_transform,
            crop,
            transforms.ToTensor(),            # convert to Tensor
            transforms.Normalize(             # normalize using ImageNet mean and std (this makes sense when using a pretrained model)
                mean=[0.485, 0.456, 0.406],   
                std=[0.229, 0.224, 0.225]
            )
        ])

        image_tensor = preprocess(self.image)
        if unsqueeze:
            return image_tensor.unsqueeze(0).to(self.device) # NOTE: unsqueezing is only helpful if passing one image at a time
        return image_tensor.to(self.device)
        
    
    
    def get_image_embedding(self, embedding_model, random_crop=False):
        embedding_model.to(self.device)
        embedding_model.fc = nn.Identity()  # Set the classification layer to identity to output embeddings
        embedding_model.eval()
        with torch.no_grad():
            # Move the input tensor to the device
            self.device = str(next(embedding_model.parameters()).device) # this way to_tensor() puts on same device as the embedding model
            embedding = embedding_model(self.to_tensor(random_crop=random_crop))
            embedding = embedding.squeeze(0).cpu().numpy()  # Correctly handle the shape
        self.embedding = embedding
        return embedding
    

    def get_fog_val(self, model, embedding_model=None, embedding=None, random_crop=False):

        model.to(self.device)
        # NOTE: model should be in evaluation mode
        # if isinstance(model, BaseEstimator): # this means sklearn classifier was passed
        if not embedding_model is None:
            embedding_model.to(self.device)
            
            if not embedding is None:
                self.embedding = embedding
            if self.embedding is None:
                # make sure we have to use embedding model

                embedding = self.get_image_embedding(embedding_model, random_crop=random_crop)
                if embedding.ndim == 1:
                    embedding = embedding.reshape(1, -1)  # Reshape if it's a single sample
                self.embedding = embedding
            # now the embedding model is set up so you can do a prediction
            
            if isinstance(model, torch.nn.Module):
                model.eval()
                with torch.no_grad():
                    embedding_tensor = torch.tensor(self.embedding, dtype=torch.float32).to(self.device)
                    logits = model(embedding_tensor) # the second output is the embedding
                    probabilities = torch.sigmoid(logits) # NOTE dont really care about probabilities for now maybe use them later on
                    self.probabilities = probabilities
                    self.logits = logits
                    # Convert probabilities to binary prediction (0 or 1)
                    self.fog_val = int(torch.argmax(logits, dim=1).item())
            else:
                # this would be an sklearn model
                self.fog_val = model.predict(embedding)[0] # return is a list of len 1 (faster to predict all together most likely)
        else:
            # end to end model that takes in image
            model.eval()
            with torch.no_grad():
                logits, _ = model(self.to_tensor()) # the second output is the embedding
                probabilities = torch.sigmoid(logits)
            
                # Convert probabilities to binary prediction (0 or 1)
                self.fog_val = int((probabilities > 0.5).float())
        return self.fog_val


    def get_fog_val_multiple_regions(self, model, embedding_model):
        """
        Tiles the image with non-overlapping crops of size `self.crop_size` (or full size if None),
        extracts embeddings for each crop, and does classification with max pooling over logits.
        """
        model.to(self.device)
        embedding_model.to(self.device)
        model.eval()
        embedding_model.eval()

        img_tensor = self.to_tensor().to(self.device)  # shape: (C, H, W)
        _, H, W = img_tensor.shape
        crop_size = self.crop_size or min(H, W)

        crops = []

        for top in range(0, H, crop_size):
            for left in range(0, W, crop_size):
                h_end = min(top + crop_size, H)
                w_end = min(left + crop_size, W)

                # If it's a small remainder patch, pad it up to crop_size
                pad_bottom = crop_size - (h_end - top)
                pad_right = crop_size - (w_end - left)

                crop = img_tensor[:, top:h_end, left:w_end]
                if pad_bottom > 0 or pad_right > 0:
                    crop = F.pad(crop, [0, pad_right, 0, pad_bottom])  # [left, right, top, bottom]

                crops.append(crop)

        crops_tensor = torch.stack(crops).to(self.device)  # shape: (num_crops, C, crop_size, crop_size)

        with torch.no_grad():
            embeddings = embedding_model(crops_tensor)  # (num_crops, 2048)
            logits = model(embeddings)  # (num_crops, 1)

            max_logit, _ = torch.max(logits, dim=0)  # max over crops
            self.logits = max_logit
            self.probabilities = torch.sigmoid(max_logit)
            print(self.probabilities)
            print(int(torch.argmax(self.probabilities).item()))

            self.fog_val = int(torch.argmax(self.probabilities).item())

        return self.fog_val


    def plot_image(self, plot_crop=False):
        image_data = self.to_numpy()
        plt.imshow(image_data)

        plt.title(f'{os.path.basename(self.filepath)}: \n {self.nocturnal=}, {self.fog_val=}')

        if plot_crop:
            # plot red box of crop
            height, width = image_data.shape[:2]
            box_size = self.crop_size
            box_x = (width - box_size) // 2  # Top-left x-coordinate
            box_y = (height - box_size) // 2  # Top-left y-coordinate
            rect = patches.Rectangle(
                (box_x, box_y),  # Bottom-left corner
                box_size,        # Width
                box_size,        # Height
                linewidth=2,     # Line thickness
                edgecolor='red', # Color
                facecolor='none' # Transparent fill
            )
            plt.gca().add_patch(rect)

        # plt.title(f"Timestamp: {self.timestamp}\nDoN: {self.nocturnal}\nFog Value: {self.fog_val}\nproba: {self.probabilities}")
        plt.show()