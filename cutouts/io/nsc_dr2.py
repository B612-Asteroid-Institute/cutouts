import logging

import numpy as np
import pandas as pd
from pyvo.dal.sia import SIAResults

from .sia import SIAHandler
from .types import CutoutRequest, CutoutResult
from .util import exposure_id_from_url

logger = logging.getLogger(__name__)


def _get_generic_image_url_from_cutout_url(cutout_url: str):
    """ """
    return cutout_url.split("&POS=")[0]


def find_cutout_nsc_dr2(
    cutout_request: CutoutRequest,
) -> CutoutResult:
    """
    Search the NOIRLab Science Archive for a cutout at a given RA, Dec, and MJD [UTC].
    """
    logger.info(
        f"Fetching NSC cutout with ra: {cutout_request.ra_deg} dec: {cutout_request.dec_deg} exposure start mjd: {cutout_request.exposure_start_mjd}"
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
    results.columns.rename(
        {
            "obs_id": "observation_id",
            "access_url": "cutout_url",
            "exptime": "exposure_duration",
            "s_ra": "ra_deg",
            "s_dec": "dec_deg",
            "mjd_obs": "exposure_start",
            "height": "height_arcsec",
            "width": "width_arcsec",
        },
        inplace=True,
    )

    results["image_url"] = results["cutout_url"].apply(
        _get_generic_image_url_from_cutout_url
    )

    # Filter out results that don't match the observation time
    results = results[
        np.abs(results["exposure_start"] - cutout_request.exposure_start_mjd)
        < cutout_request.delta_time
    ]

    results["exposure_id"] = results["cutout_url"].apply(exposure_id_from_url)

    # Only include the columns we care about
    results = results[
        [
            "cutout_url",
            "dec_deg",
            "exposure_duration",
            "exposure_id",
            "exposure_start_mjd",
            "filter",
            "height_arcsec",
            "image_url",
            "ra_deg",
            "width_arcsec",
        ]
    ]

    results.reset_index(inplace=True, drop=True)

    if len(results) == 0:
        raise ValueError("No results found.")

    # For now return just the first result
    # we may want this to be more sophistocated in the future
    result = results.to_dict(orient="records")[0]
    result = CutoutResult(
        cutout_url=result["cutout_url"],
        dec_deg=result["dec_deg"],
        exposure_duration=result["exposure_duration"],
        exposure_id=result["exposure_id"],
        exposure_start_mjd=result["exposure_start_mjd"],
        filter=result["filter"],
        height_arcsec=result["height_arcsec"],
        image_url=result["image_url"],
        ra_deg=result["ra_deg"],
        request_id=cutout_request.request_id,
        width_arcsec=result["width_arcsec"],
    )

    if cutout_request.exposure_id is not None:
        if cutout_request.exposure_id != result.exposure_id:
            err = (
                f"Exposure ID {cutout_request.exposure_id} does not match any cutouts. "
                "Check that the exposure ID is correct."
            )
            raise ValueError(err)

    if cutout_request.exposure_duration is not None:
        if cutout_request.exposure_duration != result.exposure_duration:
            err = (
                f"Exposure duration {cutout_request.exposure_duration} does not match any cutouts. "
                "Check that the exposure duration is correct."
            )
            raise ValueError(err)

    return result


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
