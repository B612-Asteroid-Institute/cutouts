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

    Raises
    ------
    FileNotFoundError
        If no cutouts are found within delta_time days of the requested exposure start time.
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


@pa.check_types
def select_comparison_cutout(
    results_df: DataFrame[CutoutsResultSchema],
    cutout_result: CutoutResult,
    cutout_request: CutoutRequest,
    min_time_separation: float = 1 / 24,
    min_exposure_duration_ratio: float = 1.0,
    same_filter: bool = True,
) -> CutoutResult:
    """
    Select a comparison cutout that is at least min_time_separation days away from the
    requested cutout, has an exposure duration at least min_exposure_duration_ratio
    compared to the requested cutout, and optionally is in the same filter.

    Parameters
    ----------
    results_df : `~pandas.DataFrame`
        Dataframe with query results.
    cutout_result : `~cutouts.io.types.CutoutResult`
        Cutout result.
    cutout_request : `~cutouts.io.types.CutoutRequest`
        Cutout request.
    min_time_separation : float
        Minimum time separation in days.
    min_exposure_duration_ratio : float
        Minimum exposure duration ratio compared to requested cutout. A ratio of 1.0
        means that the comparison cutout must have at least the same exposure duration
        as the requested cutout.
    same_filter : bool
        If True, the comparison cutout must be in the same filter as the requested cutout.

    Returns
    -------
    comparison_result : `~pandas.DataFrame`
        Dataframe with comparison cutouts.

    Raises
    ------
    FileNotFoundError
        If no comparison cutouts are found.
    """
    logger.debug(
        f"Selecting comparison cutout for {cutout_request.observatory_code} with MJD "
        f"{cutout_request.exposure_start_mjd} +- {cutout_request.delta_time} days."
    )

    # Remove the cutout that was used to make the request if it is still included
    # in the results dataframe
    candidates_filtered = results_df[
        results_df["cutout_url"] != cutout_result.cutout_url
    ].copy()

    # Add delta time column
    candidates_filtered["delta_time"] = (
        cutout_request.exposure_start_mjd - candidates_filtered["exposure_start_mjd"]
    )
    candidates_filtered["abs_delta_time"] = np.abs(candidates_filtered["delta_time"])

    # Limit to cutouts that are at least min_time_separation days away from the
    # requested exposure start time
    candidates_filtered = candidates_filtered[
        candidates_filtered["abs_delta_time"] > min_time_separation
    ]
    logger.debug(
        f"Mininum time separation filter reduced comparison cutout candidates to {len(candidates_filtered)}"
    )

    # Limit to cutouts that have an exposure duration at least min_exposure_duration_ratio compared
    # to the requested cutout
    candidates_filtered = candidates_filtered[
        candidates_filtered["exposure_duration"]
        >= min_exposure_duration_ratio * cutout_result.exposure_duration
    ]
    logger.debug(
        f"Exposure duration filter reduced comparison cutout candidates to {len(candidates_filtered)}"
    )

    # Limit to cutouts that are in the same filter
    if same_filter:
        candidates_filtered = candidates_filtered[
            candidates_filtered["filter"] == cutout_result.filter
        ]
        logger.debug(
            "Limiting cutouts to the same filter reduced comparison cutout "
            f"candidates to {len(candidates_filtered)}"
        )

    if len(candidates_filtered) == 0:
        raise FileNotFoundError(
            "No comparison cutout found for cutout at "
            f"({cutout_result.exposure_start_mjd},{cutout_result.ra_deg},{cutout_result.dec_deg})."
        )

    # Sort by delta time
    candidates_filtered = candidates_filtered.sort_values(
        by="abs_delta_time", ascending=True
    )

    # Select the first cutout
    candidate = candidates_filtered.to_dict(orient="records")[0]
    candidate = CutoutResult(
        cutout_url=candidate["cutout_url"],
        exposure_duration=candidate["exposure_duration"],
        exposure_id=candidate["exposure_id"],
        exposure_start_mjd=candidate["exposure_start_mjd"],
        filter=candidate["filter"],
        height_arcsec=candidate["height_arcsec"],
        image_url=candidate["image_url"],
        ra_deg=cutout_request.ra_deg,
        dec_deg=cutout_request.dec_deg,
        request_id=cutout_request.request_id,
        width_arcsec=candidate["width_arcsec"],
    )
    return candidate
