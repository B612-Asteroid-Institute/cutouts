# cutouts: postage stamps along the trajectory of a moving object 
#### A Python package by the Asteroid Institute, a program of the B612 Foundation 
[![Python 3.7+](https://img.shields.io/badge/Python-3.7%2B-blue)](https://img.shields.io/badge/Python-3.7%2B-blue)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)  

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
ra = results["pred_ra_deg"].values
dec = results["pred_dec_deg"].values
vra = results["pred_vra_degpday"].values
vdec = results["pred_vdec_degpday"].values
obscode = results["obscode"].values
mag = results["mag"].values
filter = results["filter"].values 
mag_sigma = results["mag_sigma"].values
times = Time(results["mjd_utc"].values, scale="utc", format="mjd")
exposure_id = results["exposure_id"].values

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
    filters=filters,
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

