import logging

import numpy as np
import pandera as pa
from pandera.typing import DataFrame

from .io.types import CutoutRequest, CutoutResult, CutoutsResultSchema

logger = logging.getLogger(__file__)


@pa.check_types
def select_cutout(
    results_df: DataFrame[CutoutsResultSchema], cutout_request: CutoutRequest
) -> CutoutResult:
    """
    Select the cutout closest to the requested exposure start time +- delta_time.

    Parameters
    ----------
    results_df : `~pandas.DataFrame`
        Dataframe with query results.
    cutout_request : `~cutouts.io.types.CutoutRequest`
        Cutout request.

    Returns
    -------
    cutout_result : `~cutouts.io.types.CutoutResult`
        Cutout result.
    """
    logger.debug(
        f"Selecting cutout for {cutout_request.observatory_code} with MJD "
        f"{cutout_request.exposure_start_mjd} +- {cutout_request.delta_time} days."
    )

    # Filter by exposure start time +- delta_time
    result_df = results_df[
        np.abs(results_df["exposure_start_mjd"] - cutout_request.exposure_start_mjd)
        < cutout_request.delta_time
    ]

    if len(result_df) == 0:
        raise FileNotFoundError(
            f"No cutouts found within {cutout_request.delta_time} "
            f"days of {cutout_request.exposure_start_mjd}."
        )

    if len(result_df) > 1:
        logger.warning(
            f"Found {len(result_df)} cutouts within {cutout_request.delta_time} days of "
            f"{cutout_request.exposure_start_mjd}. Selecting the first one."
        )

    # Convert the first cutout result to a CutoutResult object
    result = result_df.to_dict(orient="records")[0]
    result = CutoutResult(
        cutout_url=result["cutout_url"],
        exposure_duration=result["exposure_duration"],
        exposure_id=result["exposure_id"],
        exposure_start_mjd=result["exposure_start_mjd"],
        filter=result["filter"],
        height_arcsec=result["height_arcsec"],
        image_url=result["image_url"],
        ra_deg=cutout_request.ra_deg,
        dec_deg=cutout_request.dec_deg,
        request_id=cutout_request.request_id,
        width_arcsec=result["width_arcsec"],
    )

    for field in [
        "exposure_id",
        "exposure_start_mjd",
        "exposure_duration",
        "ra_deg",
        "dec_deg",
        "filter",
    ]:
        request_value = getattr(cutout_request, field)
        result_value = getattr(result, field)
        if request_value is not None:
            if request_value != result_value:
                logger.warning(
                    f"Requested {field} {request_value} does not match result {result_value}"
                )

    return result
