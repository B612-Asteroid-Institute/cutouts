import logging
import os
import shutil
from typing import Optional

import pandas as pd
from astropy.utils.data import download_file

from .nsc import find_cutouts_nsc_dr2
from .skymapper import find_cutouts_skymapper_dr2
from .types import CutoutRequest
from .ztf import find_cutouts_ztf

logger = logging.getLogger(__file__)


EXPOSURE_SEARCH_MAPPING = {
    "I41": find_cutouts_ztf,
    "Q55": find_cutouts_skymapper_dr2,
    "W84": find_cutouts_nsc_dr2,
    "V00": find_cutouts_nsc_dr2,
    "695": find_cutouts_nsc_dr2,
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


def find_cutouts(cutout_request: CutoutRequest) -> pd.DataFrame:
    """
    High level function to find a cutouts at a given Observatory, RA, Dec.

    """
    search_method = EXPOSURE_SEARCH_MAPPING.get(cutout_request.observatory_code, None)
    if search_method is None:
        raise ValueError(
            f"No search method found for obscode {cutout_request.observatory_code}."
        )

    exposure_match = search_method(cutout_request)

    return exposure_match


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
    logger.info(f"Fetching {url}...")
    path = download_file(url, **kwargs)

    if out_file is not None:
        if os.path.dirname(out_file) != "":
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
        shutil.copy(path, out_file)
        path = out_file

    return path
