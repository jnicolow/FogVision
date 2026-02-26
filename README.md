# FogVision
[![Last Commit](https://img.shields.io/github/last-commit/jnicolow/FogVision)](
https://github.com/jnicolow/FogVision/commits/)
![GitHub issues](https://img.shields.io/github/issues/jnicolow/FogVision)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![GitHub commit activity](https://img.shields.io/github/commit-activity/y/jnicolow/FogVision)

FogVision is an open-source Python framework for classifying mountain trail camera imagery by fog presence. First, image embeddings are computed using a pretrained ResNet50 model, and then a classification head was trained on ~40k images from 30 sites (separate classification head for diurnal and nocturnal imagery).

FogVision is also downloadable as a package and comes with a lightweight CLI tool. It can then be utilized from the command-line to perform inferences on folders of images. 


## 1. Installation<a name="introduction"></a>
Create a virtual environment with venv
```
cd path/to/folder
python3 -m venv .venv
source .venv/bin/activate
```
The path to the folder can be any location that you wish to host FogVision on.

Then, install the repository to the folder using `pip install`
```
python -m pip install git+https://github.com/min-808/fogvision-cli
```

The package may take a few minutes to download. Once it has finished downloading, the `fogvision` command can then be run in the environment.

```
fogvision path/to/images
```

## 2. Arguments<a name="arguments"></a>

When running `fogvision`, there are some extra parameters that can be provided to the command

```
fogvision path/to/images

--crop-size (int)
--random-crop (bool)
--threshold (int)
--save-csv-to (str)
--sitename
```

- `--crop-size`: The side length (in pixels) of the square crop that is fed into the model. It controls how big the tensor is. If nothing is passed, then it chooses the largest square that fits in the image, rounded down to a multiple of 32.
- `--random-crop`: If false, the center square is always taken. If true, then a randomly positioned square is taken.
- `--threshold`: By default, the threshold is 0.5. This means that if the inference value for the image is >= 0.5, then `fog_val` will be 1, but if it's less than 0.5, then `fog_val` will be 0. This option changes the threshold of the fog_val.
- `save-csv-to`: Allows for a path to save the .csv file to, instead of the images folder.
- `--sitename`: Allows for manual setting of the site name.