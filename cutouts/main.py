import argparse
import logging
import os
import pathlib
import sys
from typing import Any, Iterable, List, Optional, Tuple, Union, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import pandera as pa
from astropy.time import Time
from pandera.typing import DataFrame
from pyvo.dal.sia import SIAService

from .io import download_cutout, find_cutout
from .io.types import CutoutRequest, CutoutRequestSchema, CutoutResult
from .plot import plot_cutouts

logger = logging.getLogger("cutouts")


@pa.check_types
def get_cutouts(
    cutout_requests: DataFrame[CutoutRequestSchema],
    out_dir: str = "~/.cutouts",
    timeout: Optional[int] = 180,
    use_cache: bool = True,
) -> Iterable[dict[str, Any]]:
    """ """

    # Get urls and metadata for each cutout
    logger.info(f"Getting cutouts for {len(cutout_requests)} requests.")

    results = []
    for record in cutout_requests.to_dict(orient="records"):
        try:
            cutout_request = CutoutRequest(**record)  # type: ignore
            result = find_cutout(cutout_request)
        except FileNotFoundError as e:
            logger.warning(e)
            result = {"error": e}

        result = dict(result)
        results.append(result)

    for result in results:
        # Don't bother trying to download url if we didn't get a
        # result
        if "error" in result:
            continue

        full_image_path, cutout_image_path = generate_local_image_paths(result)

        result["full_image_path"] = pathlib.Path(out_dir) / full_image_path
        result["cutout_image_path"] = pathlib.Path(out_dir) / cutout_image_path

        for path in [result["full_image_path"], result["cutout_image_path"]]:
            if path.exists():
                if use_cache:
                    logger.info(
                        f"{path} already exists locally and using cache, skipping"
                    )
                    continue

            try:
                download_cutout(
                    result["cutout_url"],
                    out_file=path.as_posix(),
                    cache=True,
                    pkgname="cutouts",
                    timeout=timeout,
                )
            except FileNotFoundError as e:
                result["error"] = str(e)

    return results


def generate_local_image_paths(result: dict) -> Tuple[str, str]:
    exposure_id_str = ""
    if result["exposure_id"] is not None:
        exposure_id_str = f"_expid_{result['exposure_id']}"

    cutout_path = f"cutout_{Time(result['exposure_start_mjd'], format='mjd', scale='utc').utc.isot}_ra{result['ra_deg']:.8f}_dec{result['dec_deg']:.8f}{exposure_id_str}_h{result['height_arcsec']}_w{result['width_arcsec']}.fits"
    full_image_path = f"{Time(result['exposure_start_mjd'], format='mjd', scale='utc').utc.isot}_ra{result['ra_deg']:.8f}_dec{result['dec_deg']:.8f}{exposure_id_str}_h{result['height_arcsec']}_w{result['width_arcsec']}.fits"

    return full_image_path, cutout_path


def main():
    parser = argparse.ArgumentParser(
        prog="cutouts", description="Get and plot cutouts along a trajectory."
    )
    parser.add_argument(
        "observations",
        help="File containing observations and predicted ephemerides of a moving object.",
        type=str,
    )
    parser.add_argument(
        "--out_dir",
        help="Directory where to save downloaded cutouts and the grid of plotted cutouts.",
        type=str,
        default=".",
    )
    parser.add_argument(
        "--out_file",
        help="File name (not including --out_dir) of where to save cutouts grid.",
        type=str,
        default="cutouts.jpg",
    )

    # Set root logger to log to stdout for info and above
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

    args = parser.parse_args()

    observations = pd.read_csv(args.observations, index_col=None)

    output = run_cutouts_from_precovery(
        observations, pathlib.Path(args.out_dir), pathlib.Path(args.out_file)
    )
    print(pd.DataFrame(output).to_csv(index=False))


def run_cutouts_from_precovery(
    observations: pd.DataFrame,
    out_dir: pathlib.Path = pathlib.Path("."),
    out_file: pathlib.Path = pathlib.Path("cutout.png"),
):
    cutout_requests = observations[
        [
            "obscode",
            "exposure_mjd_start",
            "pred_ra_deg",
            "pred_dec_deg",
            "exposure_id",
            "exposure_duration",
        ]
    ].copy()

    cutout_requests.reset_index()

    cutout_requests.rename(
        columns={
            "obscode": "observatory_code",
            "exposure_mjd_start": "exposure_start_mjd",
            "pred_ra_deg": "ra_deg",
            "pred_dec_deg": "dec_deg",
        },
        inplace=True,
    )

    if "height_arcsec" not in cutout_requests:
        cutout_requests["height_arcsec"] = 20.0
    if "width_arcsec" not in cutout_requests:
        cutout_requests["width_arcsec"] = 20.0
    cutout_requests["height_arcsec"].fillna(20.0, inplace=True)
    cutout_requests["width_arcsec"].fillna(20.0, inplace=True)

    if "delta_time" not in cutout_requests:
        cutout_requests["delta_time"] = 1e-8
    cutout_requests["delta_time"].fillna(1e-8, inplace=True)

    cutout_results = get_cutouts(
        cast(DataFrame[CutoutRequestSchema], cutout_requests),
        out_dir=str(out_dir),
    )

    plot_candidates = []
    for i, result in enumerate(cutout_results):
        if "error" in result:
            candidate = {
                "path": None,
                "ra": observations["ra_deg"][i],
                "dec": observations["dec_deg"][i],
                "vra": observations["pred_vra_degpday"][i],
                "vdec": observations["pred_vdec_degpday"][i],
                "mag": observations["mag"][i],
                "mag_sigma": observations["mag_sigma"][i],
                "filter": observations["filter"][i],
                "exposure_start": observations["exposure_mjd_start"][i],
                "exposure_duration": observations["exposure_duration"][i],
                "exposure_id": observations["exposure_id"][i],
            }
        else:
            candidate = {
                "path": result["cutout_image_path"],
                "ra": result["ra_deg"],
                "dec": result["dec_deg"],
                "vra": observations["pred_vra_degpday"][i],
                "vdec": observations["pred_vdec_degpday"][i],
                "mag": observations["mag"][i],
                "mag_sigma": observations["mag_sigma"][i],
                "filter": observations["filter"][i],
                "exposure_start": result["exposure_start_mjd"],
                "exposure_duration": result["exposure_duration"],
            }
        plot_candidates.append(candidate)

    plot_candidates = pd.DataFrame(plot_candidates)
    print(plot_candidates.columns)
    # Plot cutouts
    fig, ax = plot_cutouts(
        plot_candidates,
        cutout_height_arcsec=20,
        cutout_width_arcsec=20,
    )
    fig.savefig(os.path.join(out_dir, out_file), bbox_inches="tight")

    return cutout_results
