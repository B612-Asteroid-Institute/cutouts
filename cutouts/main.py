import argparse
import logging
import os
import pathlib
from typing import Any, Iterable, List, Optional, Tuple, Union, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import pandera as pa
from astropy.time import Time
from pandera.typing import DataFrame
from pyvo.dal.sia import SIAService

from .io import download_cutout, find_cutout
from .io.types import CutoutRequest, CutoutRequestSchema
from .plot import plot_cutouts

logger = logging.getLogger("cutouts")


@pa.check_types
def get_cutouts(
    cutout_requests: DataFrame[CutoutRequestSchema],
    out_dir: str = "~/.cutouts",
    timeout: Optional[int] = 180,
    use_cache: bool = True,
) -> Iterable[dict[str, Any]]:
    """
    Attempt to find cutouts by querying the given Simple Image Access (SIA)
    catalog for cutouts at each RA, Dec, and MJD [UTC]. If the exposure ID is known
    an additional check will be made to make sure the exposure ID matches the cutout.
    If it does not, then a warning will be thrown.

    Parameters
    ----------
    cutout_requests : `.types.CutoutRequestSchema`
        Dataframe containing RA, Dec, and MJD [UTC] for each cutout.

        observatory_codes : str
            Observatory code.
        exposure_start: float
            The start time of the exposure in MJD [UTC].
        ra_deg: float
            Right Ascension in degrees.
        dec_deg: float
            Declination in degrees.
        exposure_id: str, optional
            Exposure ID, if known.
        exposure_duration: float, optional
            Exposure duration in seconds.
    delta_time: float, optional
        Match on observation time to within this delta. Delta should
        be in units of days.
    height_arcsec : float, optional
        Height of the cutout in arcseconds.
    width_arcsec : float, optional
        Width of the cutout in arcseconds.
    out_dir : str, optional
        Save cutouts to this directory. If None, cutouts are saved to
        the package cache located at ~/.cutouts.
    timeout : int, optional
        Timeout in seconds before giving up on downloading cutout.
    use_cache: bool, optional
        If True, do not download cutouts that already exist in the cache.

    Returns
    -------
    cutouts : `~pandas.DataFrame`
        Dataframe containing paths to the downloaded cutouts and related metadata.

    Raises
    ------
    ValueError: If times is not an `~astropy.Time` object.
    """

    # Get urls and metadata for each cutout

    results = []
    for record in cutout_requests.to_dict(orient="records"):
        try:
            cutout_request = CutoutRequest(**record)
            result = find_cutout(cutout_request)
        except FileNotFoundError as e:
            logger.warning(e)
            result = {"error": e}

        results.append(result)

    for result in results:
        if "error" in result:
            continue
        exposure_id_str = ""
        if result["exposure_id"] is not None:
            exposure_id_str = f"_expid_{result['exposure_id']}"

        file_name = f"{Time(result['exposure_start']).utc.isot}_ra{result['ra']:.8f}_dec{result['dec']:.8f}{exposure_id_str}_h{height_arcsec}_w{width_arcsec}.fits"

        file_path = pathlib.Path(out_dir) / file_name
        if file_path.exists():
            logger.info(f"Cutout {file_name} has been previously downloaded.")
            if use_cache:
                result["file_path"] = file_path.as_posix()
                continue

        try:
            download_cutout(
                result["cutout_url"],
                out_file=file_path.as_posix(),
                cache=True,
                pkgname="cutouts",
                timeout=timeout,
            )
        except FileNotFoundError as e:
            result["error"] = str(e)

        result["file_path"] = file_path.as_posix()

    return results


def main():
    parser = argparse.ArgumentParser(
        prog="cutouts", description="Get and plot cutouts along a trajectory."
    )
    parser.add_argument(
        "observations",
        help="File containing observations and predicted ephemerides of a moving object.",
        type=str,
    )
    parser.add_argument(
        "--out_dir",
        help="Directory where to save downloaded cutouts and the grid of plotted cutouts.",
        type=str,
        default=".",
    )
    parser.add_argument(
        "--out_file",
        help="File name (not including --out_dir) of where to save cutouts grid.",
        type=str,
        default="cutouts.jpg",
    )
    args = parser.parse_args()

    observations = pd.read_csv(args.observations, index_col=None)

    cutout_requests = observations[
        [
            "obscode",
            "exposure_mjd_start",
            "pred_ra_deg",
            "pred_dec_deg",
            "exposure_id",
            "exposure_duration",
        ]
    ].copy()

    cutout_requests.reset_index()

    cutout_requests.rename(
        columns={
            "obscode": "observatory_code",
            "exposure_mjd_start": "exposure_start_mjd",
            "pred_ra_deg": "ra_deg",
            "pred_dec_deg": "dec_deg",
        },
        inplace=True,
    )

    if "height_arcsec" not in cutout_requests:
        cutout_requests["height_arcsec"] = 20.0
    if "width_arcsec" not in cutout_requests:
        cutout_requests["width_arcsec"] = 20.0
    cutout_requests["height_arcsec"].fillna(20.0, inplace=True)
    cutout_requests["width_arcsec"].fillna(20.0, inplace=True)

    if "delta_time" not in cutout_requests:
        cutout_requests["delta_time"] = 1e-8
    cutout_requests["delta_time"].fillna(1e-8, inplace=True)

    cutout_results = get_cutouts(
        cast(DataFrame[CutoutRequestSchema], cutout_requests),
        out_dir=args.out_dir,
    )

    plot_candidates = []
    for i, result in enumerate(cutout_results):
        candidate = {
            "path": result["file_path"],
            "ra": result["ra_deg"],
            "dec": result["dec_deg"],
            "vra": observations["pred_vra_degpday"][i],
            "vdec": observations["pred_vdec_degpday"][i],
            "mag": observations["mag_sigma"][i],
            "mag_sigma": observations["mag_sigma"][i],
            "filter": observations["filter"][i],
            "exposure_start": result["exposure_mjd_start"][i],
            "exposure_duration": result["exposure_duration"][i],
        }
        plot_candidates.append(candidate)

    plot_candidates = pd.DataFrame(plot_candidates)

    # Plot cutouts
    fig, ax = plot_cutouts(
        plot_candidates,
        cutout_height_arcsec=20,
        cutout_width_arcsec=20,
    )
    fig.savefig(os.path.join(args.out_dir, args.out_file), bbox_inches="tight")
