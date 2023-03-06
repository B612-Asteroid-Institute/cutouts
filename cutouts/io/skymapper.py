import logging
from typing import Optional

import pandas as pd

from .sia import SIAHandler
from .types import CutoutRequest, CutoutResult

logger = logging.getLogger(__name__)


def find_cutout_skymapper(
    cutout_request: CutoutRequest,
) -> CutoutResult:
    """
    Search the Skymapper SIA service for a cutout at a given RA, Dec, and MJD [UTC].
    """
    logger.info(
        f"Fetching Skymapper cutout with ra: {cutout_request.ra_deg} dec: {cutout_request.dec_deg} exposure start mjd: {cutout_request.exposure_start_mjd}"
    )

    sia = Skymapper_SIA()
    results = sia.search(
        cutout_request.ra_deg,
        cutout_request.dec_deg,
        cutout_request.height_arcsec,
        cutout_request.width_arcsec,
    )
    results = pd.DataFrame(results)

    # Normalize the image url column
    results["cutout_url"] = results["get_fits"]

    # TODO: calculate a larger cutout since we can't get the whole
    # image from the SIA service.
    results["image_url"] = ""

    results = results[
        [
            "filter",
            "cutout_url",
            "exposure_start_mjd",
            "exposure_id",
            "exposure_duration",
            "image_url",
            "ra_deg",
            "dec_deg",
            "height_arcsec",
            "width_arcsec",
        ]
    ]

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
