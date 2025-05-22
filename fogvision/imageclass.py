"""
This module contains code to create and image class

Author: Joel Nicolow, Information and Computer Science, University of Hawaii at Manoa (September 24, 2024)
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import datetime

from PIL import Image


from fogvision import supportingfunctions

class CamImage(object):
    def __init__(self, filepath):
        self.filepath = filepath
        self.image = self.load_image()
        self.timestamp = self.get_timestamp()
        self.size = self.image_size()


    def __repr__(self):
        return f"ImageClass(filepath={self.filepath}, timestamp={self.timestamp})"
    

    def load_image(self):
        return supportingfunctions.load_image_PIL(self.filepath)
    

    def image_size(self):
        return self.image.size
    

    def get_timestamp(self):
        # this does not set self.timestamp in this function but is used to set it in __init__()
        timestamp = supportingfunctions.get_timestamp(image=self.image)
        if timestamp is None:
            # this could be because the image is None or because there is no metadata for the image
            try:
                timestamp = datetime.datetime.strptime(str(os.path.basename(self.filepath).replace('.JPG', '').replace('.jpg', '')), "%Y-%m-%d %H_%M_%S") # this assumes the file names are in this format
                print(timestamp)
            except ValueError:
                # this means its just not the right format and there is nothing we can do
                timestamp = None
        self.timestamp = timestamp
        return timestamp



    def to_numpy(self):
        return np.array(self.image)


    def plot_image(self):
        plt.imshow(self.to_numpy())
        plt.title(f"Timestamp: {self.timestamp}")
        plt.show()
