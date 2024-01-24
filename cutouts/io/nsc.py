import logging

import pandas as pd
from pyvo.dal.sia import SIAResults

from .sia import SIAHandler
from .util import exposure_id_from_url

logger = logging.getLogger(__name__)


def _get_generic_image_url_from_cutout_url(cutout_url: str):
    """ """
    return cutout_url.split("&POS=")[0]


def find_cutouts_nsc_dr2(
    cutout_request: pd.DataFrame,
) -> pd.DataFrame:
    """
    Search the NOIRLab Archive for cutouts and images at a given RA, Dec.

    Parameters
    ----------
    cutout_request : CutoutRequest
        The cutout request.

    Returns
    -------
    cutout_results : pd.DataFrame
        The cutout results for this area of the sky.
        Columns:
            ra_deg, dec_deg, filter, exposure_id, exposure_start_mjd, exposure_duration,
            cutout_url, image_url, height_arcsec, width_arcsec
    """
    logger.info(
        f"Fetching NSC cutout with ra: {cutout_request.ra_deg} dec: {cutout_request.dec_deg}."
    )

    sia_handler = NSC_DR2_SIA()
    results = sia_handler.search(
        cutout_request.ra_deg,
        cutout_request.dec_deg,
        cutout_request.height_arcsec,
        cutout_request.width_arcsec,
    )
    results = results.to_table().to_pandas()

    # Only include image type results.
    results = results[results["prodtype"] == "image"]
    # Rename columns to match the cutout schema
    results.rename(
        columns={
            "obs_id": "observation_id",
            "access_url": "cutout_url",
            "exptime": "exposure_duration",
            "s_ra": "ra_deg",
            "s_dec": "dec_deg",
            "mjd_obs": "exposure_start_mjd",
            "obs_bandpass": "filter",
        },
        inplace=True,
    )

    results["image_url"] = results["cutout_url"].apply(
        _get_generic_image_url_from_cutout_url
    )

    results["exposure_id"] = results["cutout_url"].apply(exposure_id_from_url)
    results.reset_index(inplace=True, drop=True)

    # Populate the height and results from the request not
    # the results from the query
    results["height_arcsec"] = cutout_request.height_arcsec
    results["width_arcsec"] = cutout_request.width_arcsec

    # If filter is "VR DECam c0007 6300.0 2600.0" change it to
    # VR. This appears to be a bug in the NSC DR2 SIA service.
    buggy_filter = "VR DECam c0007 6300.0 2600.0"
    num_buggy_filter = len(results[results["filter"] == buggy_filter])
    if num_buggy_filter > 0:
        logger.warning(
            f"Found {num_buggy_filter} instances of {buggy_filter} filter. Changing to VR."
        )
        results.loc[results["filter"] == buggy_filter, "filter"] = "VR"

    # Only include the columns we care about
    results = results[
        [
            "ra_deg",
            "dec_deg",
            "filter",
            "exposure_id",
            "exposure_start_mjd",
            "exposure_duration",
            "cutout_url",
            "image_url",
            "height_arcsec",
            "width_arcsec",
        ]
    ]

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
            format="all",
            center="overlaps",
        )
        return results
