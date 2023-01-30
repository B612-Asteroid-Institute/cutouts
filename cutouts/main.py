import os
import logging
import argparse
import numpy as np
import numpy.typing as npt
import pandas as pd

from astropy.time import Time
from typing import (
    List,
    Optional,
    Union,
    Tuple
)
from pyvo.dal.sia import SIAService

from .io import (
    find_cutout,
    download_cutout
)
from .plot import plot_cutouts

logger = logging.getLogger("cutouts")

SIA_URL = "https://datalab.noirlab.edu/sia/nsc_dr2"

def get_cutouts(
        times: Time,
        ra: npt.NDArray[np.float64],
        dec: npt.NDArray[np.float64],
        sia_url: str = SIA_URL,
        exposure_id: Optional[str] = None,
        delta_time: float = 1e-8,
        height: float = 20.,
        width: float = 20.,
        out_dir: Optional[str] = None,
        timeout: Optional[int] = 180,
    ) -> Tuple[List[Union[str, None]], pd.DataFrame]:
    """
    Attempt to find cutouts by querying the given Simple Image Access (SIA)
    catalog for cutouts at each RA, Dec, and MJD [UTC]. If the exposure ID is known
    an additional check will be made to make sure the exposure ID matches the cutout.
    If it does not, then a warning will be thrown.

    Parameters
    ----------
    times : `~astropy.time.core.Time` (N)
        Observation times.
    ra : `~numpy.ndarray` (N)
        Right Ascension in degrees.
    dec :`~numpy.ndarray` (N)
        Declination in degrees.
    sia_url : str
        Simple Image Access (SIA) service URL.
    exposure_id: str, optional
        Exposure ID, if known.
    delta_time: float, optional
        Match on observation time to within this delta. Delta should
        be in units of days.
    height : float, optional
        Height of the cutout in arcseconds.
    width : float, optional
        Width of the cutout in arcseconds.
    out_dir : str, optional
        Save cutouts to this directory. If None, cutouts are saved to
        the package cache located at ~/.cutouts.
    timeout : int, optional
        Timeout in seconds before giving up on downloading cutout.

    Returns
    -------
    list : str
        Paths of the downloaded cutouts, if no cutout was found for a particular
        position and time then None instead.
    results : `~pandas.DataFrame`
        DataFrame containing SIA search results for each cutout.

    Raises
    ------
    ValueError: If times is not an `~astropy.Time` object.
    """
    # Connect to Simple Image Access catalog
    sia_service = SIAService(sia_url)

    if not isinstance(times, Time):
        err = (
            "times should be an astropy.time object"
        )
        raise ValueError(err)

    if not isinstance(exposure_id, (List, np.ndarray)):
        exposure_id = [None for i in range(len(ra))]

    mjd = times.utc.mjd

    urls = []
    results = []
    for i, (ra_i, dec_i, mjd_i, exposure_id_i) in enumerate(zip(ra, dec, mjd, exposure_id)):

        try:
            cutout_url, results_i = find_cutout(
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
            results_i = pd.DataFrame({"access_url" : [None]})

        urls.append(cutout_url)
        results.append(results_i)

    results = pd.concat(
        results,
        ignore_index=True
    )

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
            if not os.path.exists(out_file_i):
                logger.info(f"Cutout {file_name} has been previously downloaded.")
                try:
                    path_i = download_cutout(
                        url_i,
                        out_file=out_file_i,
                        cache=True,
                        pkgname="cutouts",
                        timeout=timeout,
                    )
                except FileNotFoundError:
                    path_i = None
            else:
                path_i = out_file_i

        paths.append(path_i)

    return paths, results

def main():

    parser = argparse.ArgumentParser(
        prog="cutouts",
        description="Get and plot cutouts along a trajectory."
    )
    parser.add_argument(
        "observations",
        help="File containing observations and predicted ephemerides of a moving object.",
        type=str
        )
    parser.add_argument(
        "--out_dir",
        help="Directory where to save downloaded cutouts and the grid of plotted cutouts.",
        type=str,
        default="."
        )
    parser.add_argument(
        "--out_file",
        help="File name (not including --out_dir) of where to save cutouts grid.",
        type=str,
        default="cutouts.jpg"
    )
    parser.add_argument(
        "--sia_url",
        help="Simple Image Access (SIA) URL.",
        type=str,
        default=SIA_URL
        )
    args = parser.parse_args()


    observations = pd.read_csv(args.observations, index_col=False)

    ra = observations["pred_ra_deg"].values
    obj_ra= observations['ra_deg'].values
    dec = observations["pred_dec_deg"].values
    obj_dec = observations["dec_deg"].values
    vra = observations["pred_vra_degpday"].values
    vdec = observations["pred_vdec_degpday"].values
    times = Time(observations["mjd_utc"].values, scale="utc", format="mjd")

    if "mag" in observations.columns:
        mag = observations["mag"].values
    else:
        mag = None

    if "mag_sigma" in observations.columns:
        mag_sigma = observations["mag_sigma"].values
    else:
        mag_sigma = None

    if "filter" in observations.columns:
        filters = observations["filter"].values
    else:
        filters = None

    if "exposure_id" in observations.columns:
        exposure_id = observations["exposure_id"].values
    else:
        exposure_id = None

    cutout_paths, cutout_results = get_cutouts(
        times, ra, dec,
        sia_url=args.sia_url,
        exposure_id=exposure_id,
        out_dir=args.out_dir
    )
    exposure_time = cutout_results["exptime"].values.astype(int)

    # Plot cutouts
    fig, ax = plot_cutouts(
        cutout_paths,
        times,
        ra,
        dec,
        obj_ra,
        obj_dec,
        vra,
        vdec,
        filters=filters,
        mag=mag,
        mag_sigma=mag_sigma,
        exposure_time=exposure_time,
        cutout_height=75,
        cutout_width=75,
    )
    fig.savefig(os.path.join(args.out_dir, args.out_file), bbox_inches="tight")


