import logging
import os
import shutil
from typing import Optional
from urllib.error import HTTPError

import pandas as pd
from astropy.utils.data import download_file

from .nsc_dr2 import find_cutout_nsc_dr2
from .skymapper import find_cutout_skymapper
from .types import CutoutRequest, CutoutResult
from .ztf import find_cutout_ztf

logger = logging.getLogger(__file__)


EXPOSURE_SEARCH_MAPPING = {
    "I41": find_cutout_ztf,
    "OTHER": find_cutout_nsc_dr2,
    "HUH": find_cutout_skymapper,
}


def filter_by_exposure_id(
    exposure_matches: pd.DataFrame, exposure_id: str
) -> pd.DataFrame:
    """
    Filter the exposure matches by the exposure ID.

    Parameters
    ----------
    exposure_matches : `~pandas.DataFrame`
        Dataframe with SIA query results.
    exposure_id : str
        Exposure ID.

    Returns
    -------
    exposure_matches : `~pandas.DataFrame`
        Dataframe with SIA query results.
    """
    exposure_matches = exposure_matches[
        exposure_matches["exposure_id"].str.contains(exposure_id)
    ]
    return exposure_matches


def verify_exposure_duration(
    exposure_matches: pd.DataFrame, exposure_duration: float
) -> None:
    """
    Verify that the exposure duration matches the cutout.

    Parameters
    ----------
    exposure_matches : `~pandas.DataFrame`
        Dataframe with SIA query results.
    exposure_duration : float
        Exposure duration in seconds.

    Raises
    ------
    ValueError: If the exposure duration does not match the cutout.
    """
    if not exposure_matches["exposure_duration"] == exposure_duration:
        err = (
            f"Exposure duration {exposure_duration} does not match any cutouts. "
            "Check that the exposure duration is correct."
        )
        logger.warning(err)


def find_cutout(cutout_request: CutoutRequest) -> CutoutResult:
    """
    High level function to find a cutout at a given Observatory, RA, Dec, and MJD [UTC].

    Parameters
    ----------
    obscode : str
        Observatory code. This determines which backend to use.
    ra_deg : float
        Right Ascension in degrees.
    dec_deg : float
        Declination in degrees.
    exposure_start : float
        The start time of the exposure in MJD [UTC].
    delta_time: float, optional
        Match on observation time to within this delta. Delta should
        be in units of days.
    height_arcsec : float, optional
        Height of the cutout in arcseconds.
    width_arcsec : float, optional
        Width of the cutout in arcseconds.
    exposure_id: str, optional
        Exposure ID, if known.
    exposure_duration: float, optional

    Returns
    -------
    result : dict
        The best matched result from the query
        url: str
            URL of remote cutout.
        exposure_start_mjd: float
            Exposure start time in MJD [UTC].
        exposure_id: str
            Exposure ID.
        exposure_duration: float
            Exposure duration in seconds.
        ra_deg: float
            Right Ascension in degrees.
        dec_deg: float
            Declination in degrees.
        height_arcsec: float
            Height of the cutout in arcseconds.
        width_arcsec: float
            Width of the cutout in arcseconds.

    Raises
    ------
    FileNotFoundError: If no cutout is found at the given RA, Dec, MJD [UTC]
        using this particular SIA Service.
    """
    search_method = EXPOSURE_SEARCH_MAPPING.get(cutout_request.observatory_code, None)
    if search_method is None:
        raise ValueError(
            f"No search method found for obscode {cutout_request.observatory_code}."
        )

    exposure_matches = search_method(cutout_request)

    if len(exposure_matches) == 0:
        err = "No cutout found."
        raise FileNotFoundError(err)

    # if exposure_id is not None:
    #     exposure_matches = filter_by_exposure_id(exposure_matches, exposure_id)

    if len(exposure_matches) == 0:
        raise Exception(
            f"No results matched provided exposure ID {cutout_request.exposure_id}."
        )

    # For now, we just take the first result.
    # Later we may want to expose an API to return all the reults
    result = exposure_matches.iloc[:1].to_dict("records")[0]

    if cutout_request.exposure_duration is not None:
        if cutout_request.exposure_duration != result["exposure_duration"]:
            err = (
                f"Exposure duration {cutout_request.exposure_duration} does not match any cutouts. "
                "Check that the exposure duration is correct."
            )
            raise ValueError(err)

    result = CutoutResult(**result)

    return result


def download_cutout(url: str, out_file: Optional[str] = None, **kwargs) -> str:
    """
    Download cutout located at url. This function
    uses `~astropy.utils.data.download_file`.

    Parameters
    ----------
    url : str
        URL of remote cutout.
    out_file : str, optional
        Save cutout to out_file.
    **kwargs : dict
        Additional keyword arguments to pass to
        `~astropy.utils.data.download_file`.

    Returns
    -------
    path : str
        Location of downloaded cutout.
    """
    try:
        path = download_file(url, **kwargs)
    except HTTPError as e:
        raise FileNotFoundError(str(e))

    if out_file is not None:
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        shutil.copy(path, out_file)
        path = out_file

    return path
