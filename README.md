# cutouts: postage stamps along the trajectory of a moving object 
#### A Python package by the Asteroid Institute, a program of the B612 Foundation 
[![Python 3.7+](https://img.shields.io/badge/Python-3.7%2B-blue)](https://img.shields.io/badge/Python-3.7%2B-blue)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![DOI](https://zenodo.org/badge/480927468.svg)](https://zenodo.org/badge/latestdoi/480927468)  
[![Python Package with conda](https://github.com/B612-Asteroid-Institute/cutouts/actions/workflows/python-package-conda.yml/badge.svg)](https://github.com/B612-Asteroid-Institute/cutouts/actions/workflows/python-package-conda.yml)
[![Publish Python Package to conda](https://github.com/B612-Asteroid-Institute/cutouts/actions/workflows/python-publish-conda.yml/badge.svg)](https://github.com/B612-Asteroid-Institute/cutouts/actions/workflows/python-publish-conda.yml)  
[![Anaconda-Server Badge](https://anaconda.org/asteroid-institute/cutouts/badges/version.svg)](https://anaconda.org/asteroid-institute/cutouts)
[![Anaconda-Server Badge](https://anaconda.org/asteroid-institute/cutouts/badges/platforms.svg)](https://anaconda.org/asteroid-institute/cutouts)
[![Anaconda-Server Badge](https://anaconda.org/asteroid-institute/cutouts/badges/downloads.svg)](https://anaconda.org/asteroid-institute/cutouts)  

## Installation 

### Conda

To get the latest released version and install it into a conda environment:  
`conda install -c asteroid-institute cutouts`  

### Source

To install the bleeding edge source code, clone this repository and then:  

If you use conda to manage packages and environments:  
`conda install -c defaults -c conda-forge --file requirements.txt`  
`pip install . --no-deps`  

If you would rather download dependencies with pip:  
`pip install -r requirements.txt`  
`pip install . --no-deps`  

## Example 
To reproduce the example that ships with the repository:
```
import pandas as pd
import matplotlib.pyplot as plt
from astropy.time import Time
from cutouts import get_cutouts
from cutouts import plot_cutouts

# Read in observations and predicted ephemerides
observations = pd.read_csv("examples/2013_RR163.csv", index_col=False)

# Extract relevant quantities
ra = observations["pred_ra_deg"].values
dec = observations["pred_dec_deg"].values
vra = observations["pred_vra_degpday"].values
vdec = observations["pred_vdec_degpday"].values
obscode = observations["obscode"].values
mag = observations["mag"].values
filter = observations["filter"].values 
mag_sigma = observations["mag_sigma"].values
times = Time(observations["mjd_utc"].values, scale="utc", format="mjd")
exposure_id = observations["exposure_id"].values

# Find cutouts and save them
NSC_SIA_URL = "https://datalab.noirlab.edu/sia/nsc_dr2"
cutout_paths, cutout_results = get_cutouts(
    times, ra, dec, 
    sia_url=NSC_SIA_URL, 
    exposure_id=exposure_id, 
    out_dir="examples/cutouts"
)
exposure_time = cutout_results["exptime"].values.astype(int)

# Plot cutouts
fig, ax = plot_cutouts(
    cutout_paths, 
    times, 
    ra, dec, 
    vra, vdec, 
    filters=filter,
    mag=mag, 
    mag_sigma=mag_sigma, 
    exposure_time=exposure_time,
    cutout_height=75, 
    cutout_width=75,
)
fig.suptitle(f"(2013 RR163)", y=1.0)
fig.savefig(f"examples/2013_RR163.jpg", bbox_inches="tight")
```
![2013 RR163) Cutouts Example](examples/2013_RR163.jpg "(2013 RR163) Cutouts Example")  

`cutouts` also comes with a simple command line interface (CLI) to quickly produce a grid of cutouts from a file 
of predicted ephemerides and observations.  
```
cutouts examples/2013_RR163.csv --out_dir examples/cli_test
```
