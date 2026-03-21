# FogVision
[![Last Commit](https://img.shields.io/github/last-commit/jnicolow/FogVision)](
https://github.com/jnicolow/FogVision/commits/)
![GitHub issues](https://img.shields.io/github/issues/jnicolow/FogVision)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![GitHub commit activity](https://img.shields.io/github/commit-activity/y/jnicolow/FogVision)

FogVision is an open-source Python framework for classifying mountain trail camera imagery by fog presence. First, image embeddings are computed using a pretrained ResNet50 model, and then a classification head was trained on ~40k images from 30 sites (separate classification head for diurnal and nocturnal imagery).

FogVision can be installed either at the Python Package Index (PyPI) or through the command line. It can then be utilized, either with Jupyter Notebook or from the command-line to perform inferences on images.

## Installation
The source code can be found on GitHub at: https://github.com/jnicolow/FogVision/

FogVision is installable at the [Python Package Index (PyPI)](https://pypi.org/project/FogVision/).
```
pip install fogvision
```

## Classify function
When using FogVision, you can use the `classify` function in a .ipynb file (Jupyter Notebook) to classify your images.
```
from fogvision import fv

fv.classify(image_folder, plot_image=False, save_csv_to=None, sitename=None, crop_size=None, random_crop=False, threshold=0.5)
```
Only *image_folder* is needed to run the function properly.
- **image_folder** (str): The folder path which contains the images to be classified.
- **plot_image** (bool): Determines whether or not an image is plotted or not. Default set to false.
- **save-csv-to** (str): Allows for a path to save the .csv file to, instead of the images folder.
- **sitename**: Allows for manual setting of the site name.
- **crop-size** (int): The side length (in pixels) of the square crop that is fed into the model. It controls how big the tensor is. If nothing is passed, then it chooses the largest square that fits in the image, rounded down to a multiple of 32.
- **random-crop** (bool): If false, the center square is always taken. If true, then a randomly positioned square is taken. Default set to false.

## CLI Tool
FogVision can also be used in the command line using `fogvision`. Since images cannot be plotted in a command line, the original *plot_image* parameter is not included and set to false by default.
```
fogvision path/to/images

--save-csv-to (str)
--sitename
--crop-size (int)
--random-crop (bool)
```