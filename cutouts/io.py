import os
import shutil
import logging
import pandas as pd
from pyvo.dal.sia import SIAService
from typing import (
    Tuple,
    Optional
)
from astropy.utils.data import download_file

logger = logging.getLogger(__file__)

def exposure_id_from_url(
        url: str,
        preamble: str = "siaRef=",
        postamble: str = ".fits.fz"
    ) -> str:
    """
    Attempt to determine the exposure ID from a cutout URL.

    Parameters
    ----------
    url : str
        URL to remote cutout.
    preamble : str, optional
        URL component expected directly infront of the exposure ID.
    postamble : str, optional
        URL component expected directly after the exposure ID. This
        is sometimes the file extension.

    Returns
    -------
    exposure_id : str
        Exposure ID read from URL.
    """
    id_start = url.find(preamble)
    id_end = url.find(postamble)
    exposure_id = url[id_start + len(preamble) : id_end]
    return exposure_id


def find_cutout(
        ra: float,
        dec: float,
        mjd_utc: float,
        sia_service: SIAService,
        delta_time: float = 1e-8,
        height: float = 20,
        width: float = 20,
        exposure_id: Optional[str] = None,
    ) -> Tuple[str, pd.DataFrame]:
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
    sia_service : `~pyvo.dal.sia.SIAService`
        Simple Image Access (SIA) service to query for cutout.
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
        Dataframe with SIA query results.

    Raises
    ------
    FileNotFoundError: If no cutout is found at the given RA, Dec, MJD [UTC]
        using this particular SIA Service.
    """
    center = (ra, dec)
    result = sia_service.search(center, size=(height/3600., width/3600.)).to_table().to_pandas()
    if "mjd_obs" in result.columns:
        mjd_utcs = result["mjd_obs"].values.astype(float)
    else:
        mjd_utcs = result["mjd_utc_obs"].values.astype(float)

    logger.info(f"SIA query returned table with {len(result)} row.")
    result = result[(mjd_utcs <= mjd_utc + delta_time) & (mjd_utcs >= mjd_utc - delta_time) & (result["prodtype"] == "image")]
    result.reset_index(inplace=True, drop=True)

    logger.info(f"Filtering on {mjd_utc} +- {delta_time} MJD [UTC] reduces table to {len(result)} row(s).")
    if len(result) == 0:
        err = ("No cutout found.")
        raise FileNotFoundError(err)

    cutout_url = result["access_url"].values[0]
    if exposure_id is not None:
        url_exposure_id = exposure_id_from_url(cutout_url)

        if exposure_id != url_exposure_id:
            logger.warning(
                f"Exposure ID ({url_exposure_id}) found via search on RA, Dec," \
                f"and MJD [UTC] does not match the given exposure ID ({exposure_id})."
            )

    return cutout_url, result

def download_cutout(
        url: str,
        out_file: Optional[str] = None,
        **kwargs
    ) -> str:
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
    path = download_file(
        url,
        **kwargs
    )
    if out_file is not None:
        os.makedirs(
            os.path.dirname(out_file),
            exist_ok=True
        )
        shutil.copy(path, out_file)
        path = out_file

    return path
