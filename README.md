# CoastVision
[![Last Commit](https://img.shields.io/github/last-commit/jnicolow/FogVision)](
https://github.com/jnicolow/FogVision/commits/)
![GitHub issues](https://img.shields.io/github/issues/jnicolow/FogVision)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![GitHub commit activity](https://img.shields.io/github/commit-activity/y/jnicolow/FogVision)

FogVision is an open-source Python framework for classifying mountain trail camera imagery by fog presense.

CoastVision is inspired by [CoastSat](https://github.com/kvos/CoastSat) and [CoastSat.PlanetScope](https://github.com/ydoherty/CoastSat.PlanetScope) with the key differences being CoastVision's inclusion of an API to download PlanetScope imagery and how shoreline contours are extracted. CoastSat.PlanetScope classifies each pixel as white-water, water, sand, or other land; then using a thresholding algorithm such as peak fraction on the normalized difference water index. CoastVision classifies pixels as either land or water and then uses the marching squares algorithm to delineate a shoreline between the land and water classes.

<img src="media/stages_plot.jpg" alt="Stages Plot">


### Table of Contents

- [Installation](#installation)
- [PlanetScope API](#api)
- [Image Co-registration](#coreg)
- [Shoreline Extraction](#sds)
   - [Image Segmentation](#seg)
   - [Shoreline Extraction](#shoreline)
   - [Transect Intersection](#intersect)
- [Tidal Corrections](#tide)
- [QAQC](#qaqc)



## 1. Installation<a name="introduction"></a>
Use `coastvision.yml` to create conda environment. This will take a few minutes.
```
cd path/to/CoastVision
conda env create -f coastvision.yml
conda activate coastvision
```

After successfully creating the environment run through `example_notebook.ipynb` using the `coastvision` environment. This notebook provides code and an explanation of the following steps. Given an area of interest and a timeframe of interest, you can download imagery and extract shorelines from your own site all from this notebook.

## 2. PlanetScope API<a name="api"></a>
<a href='https://developers.planet.com/docs/data/planetscope/'>PlanetScope</a> is a satellite constellation operated by <a href='https://www.planet.com/'>Planet Labs Inc.</a> The PlanetScope constellation is made up of roughly 130 satellites, capable of imaging the entire land surface of earth with daily revisit times and 3 meter spatial resolution. The imagery has four bands red, green, blue, and near-infrared. 

Given an API key, an area of interest polygon, and a timeframe, applicable imagery will be downloaded from Planet. See section 1 "Download PlanetScope Imager" in `example_notebook.ipynb` for more info.
