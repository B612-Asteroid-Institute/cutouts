import logging

import pandas as pd
import pandera as pa
from pandera.typing import DataFrame

from .sia import SIAHandler
from .types import CutoutRequest, CutoutsResultSchema

logger = logging.getLogger(__name__)


def _get_generic_image_url_from_cutout_url(cutout_url: str):
    """ """
    url_string = (
        cutout_url.split("&")[0]
        + "&size=0.17,0.17&"
        + cutout_url.split("&")[2]
        + "&format=fits"
    )
    return url_string


@pa.check_types
def find_cutouts_skymapper(
    cutout_request: CutoutRequest,
) -> DataFrame[CutoutsResultSchema]:
    """
    Search the Skymapper SIA service for cutouts and images at a given RA, Dec.

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
        f"Fetching Skymapper cutouts with ra: {cutout_request.ra_deg} dec: {cutout_request.dec_deg}."
    )

    sia = Skymapper_SIA()
    results = sia.search(
        cutout_request.ra_deg,
        cutout_request.dec_deg,
        cutout_request.height_arcsec,
        cutout_request.width_arcsec,
    )
    results = pd.DataFrame(results)
    # Limit results to just fits files
    results = results[results["format"] == "image/fits"]

    results.rename(
        columns={
            "band": "filter",
            "exptime": "exposure_duration",
            "ra_cntr": "ra_deg",
            "dec_cntr": "dec_deg",
            "mjd_obs": "exposure_start_mjd",
        },
        inplace=True,
    )

    # Normalize the cutout url column
    results["cutout_url"] = results["get_fits"]

    results["height_arcsec"] = results["size"].apply(lambda x: x[0])
    results["width_arcsec"] = results["size"].apply(lambda x: x[1])
    results["exposure_id"] = results["unique_image_id"].apply(lambda x: x.split("-")[0])

    results["image_url"] = results["cutout_url"].apply(
        _get_generic_image_url_from_cutout_url
    )

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


class Skymapper_SIA(SIAHandler):
    SIA_URL = "https://api.skymapper.nci.org.au/public/siap/dr2/query?"

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
