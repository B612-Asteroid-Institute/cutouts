import io
import logging
from typing import Optional, Tuple

import pandas as pd
import requests

from .types import CutoutRequest

logger = logging.getLogger(__file__)


def generate_ztf_cutout_url_for_result(
    ra_deg: float,
    dec_deg: float,
    height_arcsec: float,
    width_arcsec: float,
    filefracday: float,
    field: int,
    ccdid: int,
    imgtypecode: str,
    filtercode: str,
    qid: int,
) -> str:
    """
    Generate a cutout URL from a ZTF SIA result.

    Parameters
    ----------
    result : `~pandas.DataFrame`
        Dataframe with SIA query results.

    Returns
    -------
    cutout_url : str
        URL to cutout
    """
    filefracday_str = str(filefracday)
    year = str(filefracday_str)[:4]
    monthday = str(filefracday_str)[4:8]
    fracday = str(filefracday_str)[8:]
    paddedfield = str(field).zfill(6)
    paddedccdid = str(ccdid).zfill(2)
    imgtypecode = str(imgtypecode)
    filtercode = str(filtercode)
    qid_str = str(qid)

    image_url = (
        f"https://irsa.ipac.caltech.edu/ibe/data/ztf/products/sci/"
        f"{year}/"
        f"{monthday}/"
        f"{fracday}/"
        f"ztf_{filefracday_str}_"
        f"{paddedfield}_"
        f"{filtercode}_"
        f"c{paddedccdid}_"
        f"{imgtypecode}_"
        f"q{qid_str}_"
        f"sciimg.fits"
    )

    cutout_url = f"{image_url}?center={ra_deg},{dec_deg}&size={height_arcsec},{width_arcsec}arcsec&gzip=false"
    return cutout_url


def find_cutout_ztf(cutout_request: CutoutRequest) -> pd.DataFrame:
    """
    Find cutout for a given RA, Dec, and MJD [UTC].

    Parameters
    ----------
    ra : float
        Right Ascension in degrees.
    dec : float
        Declination in degrees.
    mjd_utc : float
        Observation time in MJD [UTC].
    delta_time: float, optional
        Match on observation time to within this delta. Delta should
        be in units of days.
    height : float, optional
        Height of the cutout in arcseconds.
    width : float, optional
        Width of the cutout in arcseconds.
    exposure_id: str, optional
        Exposure ID, if known.

    Returns
    -------
    cutout_url : str
        URL to cutout
    result : `~pandas.DataFrame`
        Dataframe with image query results.

    Raises
    ------
    FileNotFoundError: If no cutout is found at the given RA, Dec, MJD [UTC]
        using this particular service.
    """

    ZTF_URL_BASE = "https://irsa.ipac.caltech.edu/ibe/search/ztf/products/sci"

    width_deg = cutout_request.width_arcsec / 3600
    height_deg = cutout_request.height_arcsec / 3600

    search_url = f"{ZTF_URL_BASE}?POS={cutout_request.ra_deg},{cutout_request.dec_deg}&SIZE={width_deg},{height_deg}&ct=csv"
    response = requests.get(search_url)
    response.raise_for_status()
    results = pd.read_csv(io.StringIO(response.text))
    results["exposure_start_mjd"] = results["obsjd"].values.astype(float) - 2400000.5
    results.rename(
        columns={
            "filtercode": "filter",
            "ra": "ra_deg",
            "dec": "dec_deg",
            "expid": "exposure_id",
            "exptime": "exposure_duration",
        },
        inplace=True,
    )

    logger.info(f"ZTF query returned table with {len(results)} row.")
    results = results[
        (
            results["exposure_start_mjd"]
            <= cutout_request.exposure_start_mjd + cutout_request.delta_time
        )
        & (
            results["exposure_start_mjd"]
            >= cutout_request.exposure_start_mjd - cutout_request.delta_time
        )
    ]
    results.reset_index(inplace=True, drop=True)

    logger.info(
        f"Filtering on {cutout_request.exposure_start_mjd} +- {cutout_request.delta_time} MJD [UTC] reduces table to {len(results)} row(s)."
    )
    if len(results) == 0:
        err = "No cutout found."
        raise FileNotFoundError(err)

    # Assign values based on the image metadata to form the url
    results["cutout_url"] = results.apply(
        lambda row: generate_ztf_cutout_url_for_result(
            ra_deg=cutout_request.ra_deg,
            dec_deg=cutout_request.dec_deg,
            height_arcsec=cutout_request.height_arcsec,
            width_arcsec=cutout_request.width_arcsec,
            filefracday=row["filefracday"],
            field=row["field"],
            ccdid=row["ccdid"],
            imgtypecode=row["imgtypecode"],
            filtercode=row["filter"],
            qid=row["qid"],
        ),
        axis=1,
    )

    results = results[
        [
            "cutout_url",
            "exposure_start_mjd",
            "ra_deg",
            "dec_deg",
            "filter",
            "exposure_duration",
        ]
    ]

    return results
