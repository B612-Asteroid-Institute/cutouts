import os
import logging
import argparse
import numpy as np

from astropy.time import Time
from typing import List
from pyvo.dal import sia

from .io import (
    find_cutout,
    download_cutout
)

logger = logging.getLogger("cutouts")

CATALOG_URL = "https://datalab.noirlab.edu/sia/nsc_dr2"

def get_cutouts(
        times,
        ra,
        dec,
        sia_catalog=CATALOG_URL,
        exposure_id=None,
        delta_time=1e-8,
        height=20,
        width=20,
        out_dir=None
    ):
    # Connect to Simple Image Access catalog
    sia_service = sia.SIAService(sia_catalog)

    if not isinstance(times, Time):
        err = (
            "times should be an astropy.time object"
        )
        raise ValueError(err)

    if not isinstance(exposure_id, (List, np.ndarray)):
        exposure_id = [None for i in range(len(ra))]

    mjd = times.utc.mjd

    urls = []
    for i, (ra_i, dec_i, mjd_i, exposure_id_i) in enumerate(zip(ra, dec, mjd, exposure_id)):

        try:
            cutout_url = find_cutout(
                ra_i,
                dec_i,
                mjd_i,
                sia_service,
                delta_time=delta_time,
                height=height,
                width=width,
                exposure_id=exposure_id_i
            )

        except FileNotFoundError as e:
            logger.warning(f"No cutout found for {mjd_i} MJD [UTC] at (RA, Dec) = ({ra_i}, {dec_i})")
            cutout_url = None

        urls.append(cutout_url)

    paths = []
    for i, (ra_i, dec_i, mjd_i, exposure_id_i, url_i) in enumerate(zip(ra, dec, mjd, exposure_id, urls)):

        if exposure_id_i is None:
            file_name = f"{times[i].utc.isot}_ra{ra_i:.8f}_dec{dec_i:.8f}_h{height}_w{width}.fits"
        else:
            file_name = f"{times[i].utc.isot}_ra{ra_i:.8f}_dec{dec_i:.8f}_expid_{exposure_id_i}_h{height}_w{width}.fits"

        if out_dir is not None:
            out_file_i = os.path.join(out_dir, file_name)
        else:
            out_file_i = None

        if url_i is None:
            path_i = None
        else:
            path_i = download_cutout(
                url_i,
                out_file=out_file_i
            )

        paths.append(path_i)

    return paths


def main():

    my_parser = argparse.ArgumentParser(
        prog="cutouts",
        description="Get and plot cutouts along a trajectory."
    )
    my_parser.parse_args()