"""
This module contains general purpose helper functions

Author: Joel Nicolow, Information and Computer Science, University of Hawaii at Manoa (September 24, 2024)
"""

import os
import pandas as pd
from PIL import Image, UnidentifiedImageError
from PIL.ExifTags import TAGS        # getting image meta data from images
import datetime as dt


def load_image_PIL(image_fn):
    # if an image path was given instead of an image load the image
    try:
        image = Image.open(image_fn)
    except UnidentifiedImageError:
        print(f'supportingfunctions.get_timestamp(): {os.path.basename(image_fn)} is a currupted image and is being skipped. Timestamp returned as None')
        return None
    return image


def get_timestamp(image:Image, getflash:bool=False):
        """
        This function gets the timestamp from an images meta data and sets self.TIMESTAMP in addition to setting the DIURNAL_NOCTURNAL mode.

        :return: self.TIMESTAMP
        """

        if not isinstance(image, Image.Image):
            # if an image path was given instead of an image load the image
            image = load_image_PIL(image)
            if image is None: return None
        
        if image._getexif() is None:
            return None # this means there is no meta data to get the timestamp from
        # image.getexif().items()
        exif = {}
        for tag, value in image._getexif().items():
            if tag in TAGS:
                exif[TAGS[tag]] = value

        dtFormat = "%Y:%m:%d %H:%M:%S"
        try:
            timestamp = dt.datetime.strptime(exif["DateTimeOriginal"], dtFormat)  # convert string from meta data to
        except KeyError:
            timestamp = None
        if getflash:
            # also, we want to get if the image uses flash or not https://github.com/ianare/exif-py/blob/develop/exifread/tags/exif.py
            DoN = 0 # default (diurnal/day)
            try:
                exif['Flash']
            except KeyError:
                DoN = 0 # default (diurnal/day)

            else:
                if exif["Flash"] == 13:
                    #  13: 'Flash fired, compulsory flash mode, return light not detected',
                    DoN = 1
                elif exif["Flash"] == 16:
                    # 16: 'Flash did not fire, compulsory flash mode'
                    DoN = 0
                elif exif["Flash"] == 32:
                    # this is for cameras that all just have 32
                    if exif['ISOSpeedRatings'] >= 3000 and exif['ExposureIndex'] >= 90:
                        DoN = 1 # this is most likely a night time photo
            return {'timestamp':timestamp, 'flash':DoN}
        return timestamp


def round_to_nearest_15(dt):
    """Rounds a datetime to the nearest 15-minute interval."""
    minutes = (dt.minute // 15) * 15
    remainder = dt.minute % 15
    if remainder >= 7.5:  # Round up if closer to next interval
        minutes += 15
    
    return dt.replace(minute=minutes % 60, second=0, microsecond=0) + pd.DateOffset(hours=(minutes // 60))
