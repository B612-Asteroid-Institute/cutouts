import numpy as np
import pandas as pd
from pyvo.dal.sia import SIAResults

from .sia import SIAHandler
from .util import exposure_id_from_url


def find_cutout_nsc_dr2(
    ra_deg: float,
    dec_deg: float,
    mjd_utc: float,
    delta_time: float = 1e-8,
    height_arcsec: float = 20,
    width_arcsec: float = 20,
) -> pd.DataFrame:
    """
    Search the NOIRLab Science Archive for a cutout at a given RA, Dec, and MJD [UTC].

    Parameters
    ----------
    ra_deg : float
        Right Ascension in degrees.
    dec_deg : float
        Declination in degrees.
    mjd_utc : float
        Observation time in MJD [UTC].
    delta_time: float, optional
        Match on observation time to within this delta. Delta should
        be in decimal units of days.
    height_arcsec : float, optional
        Height of the cutout in arcseconds.
    width_arcsec : float, optional
        Width of the cutout in arcseconds.
    exposure_id: str, optional
        Exposure ID, if known.
    exposure_duration: float, optional
        Exposure duration in seconds, if known.

    Returns
    -------
    results : `~pandas.DataFrame`
        Dataframe with SIA query results.
    """
    sia_handler = NSC_DR2_SIA()
    results = sia_handler.search(ra_deg, dec_deg, height_arcsec, width_arcsec)

    results = results.to_table().to_pandas()

    # Only include image type results.
    results = results[results["prodtype"] == "image"]

    # Rename columns to match the cutout schema
    results.columns.rename(
        {
            "obs_id": "observation_id",
            "access_url": "url",
            "exptime": "exposure_duration",
            "s_ra": "ra_deg",
            "s_dec": "dec_deg",
            "mjd_obs": "exposure_start",
            "height": "height_arcsec",
            "width": "width_arcsec",
        },
        inplace=True,
    )

    # Filter out results that don't match the observation time
    results = results[np.abs(results["exposure_start"] - mjd_utc) < delta_time]

    results["exposure_id"] = results["url"].apply(exposure_id_from_url)

    # Only include the columns we care about
    results = results[
        [
            "url",
            "exposure_start",
            "ra_deg",
            "dec_deg",
            "exposure_id",
            "exposure_duration",
            "height_arcsec",
            "width_arcsec",
        ]
    ]

    results.reset_index(inplace=True, drop=True)
    return results


class NSC_DR2_SIA(SIAHandler):
    SIA_URL = "https://datalab.noirlab.edu/sia/nsc_dr2"

    def search(
        self,
        ra_deg: float,
        dec_deg: float,
        height_arcsec: float = 20,
        width_arcsec: float = 20,
    ) -> SIAResults:
        results = self.sia_service.search(
            (ra_deg, dec_deg),
            size=(height_arcsec / 3600.0, width_arcsec / 3600.0),  # type: ignore
        )

        return results
