from typing import Optional

import pandas as pd

from .sia import SIAHandler


def find_cutout_skymapper(
    ra_deg: float,
    dec_deg: float,
    exposure_start: float,
    height_arcsec: float = 20,
    width_arcsec: float = 20,
    delta_time: float = 1e-8,
    exposure_id: Optional[str] = None,
    exposure_duration: Optional[float] = None,
) -> pd.DataFrame:
    """
    Find a cutout from Skymapper.

    Parameters
    ----------
    ra_deg : float
        Right ascension in degrees.
    dec_deg : float
        Declination in degrees.
    height_arcsec : float
        Height of cutout in arcseconds.
    width_arcsec : float
        Width of cutout in arcseconds.
    exposure_start_mjd : float
        Modified Julian date in UTC.
    delta_time : float
        Time window in days.
    exposure_id : str
        Exposure ID.
    exposure_duration : float
        Exposure duration in seconds.

    Returns
    -------
    request : dict
        Request parameters returned as a dictionary

        obscode : str
            Observatory code.
        exposure_start_mjd
            Exposure modified Julian date in UTC.
        delta_time : floatp
            Time window in days.
        ra_deg : float
            Right ascension in degrees of the observation
        dec_deg : float
            Declination in degrees of the observation
        height_arcsec : float
            Height of the cutout in arcseconds.
        width_arcsec : float
            Width of the cutout in arcseconds.
        exposure_id : str
            Requested exposure ID.
        exposure_duration : float
            Expected exposure duration in seconds.

    result : `~pandas.DataFrame`
        filter : str
            Filter name.
        cutout_url : str
            URL of remote cutout.
        exposure_start : float
            Exposure start time in MJD [UTC].
        exposure_id : str
            Exposure ID.
        exposure_duration : float
            Exposure duration in seconds.





    """
    sia = Skymapper_SIA()
    results = sia.search(ra_deg, dec_deg, height_arcsec, width_arcsec)
    results = pd.DataFrame(results)

    # Normalize the image url column
    results["cutout_url"] = results["get_fits"]

    results = results[
        ["cutout_url", "ra_deg", "dec_deg", "height_arcsec", "width_arcsec"]
    ]

    return results


class Skymapper_SIA(SIAHandler):
    SIA_URL = "https://datalab.noirlab.edu/sia/nsc_dr2"

    def search(
        self,
        ra_deg: float,
        dec_deg: float,
        height_arcsec: float = 20,
        width_arcsec: float = 20,
    ) -> pd.DataFrame:
        result = self.sia_service.search(
            (ra_deg, dec_deg), size=(height_arcsec / 3600.0, width_arcsec / 3600.0)
        )
        result = pd.DataFrame(result)

        return result
