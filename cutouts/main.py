import argparse
import logging
import pathlib
import sys
from typing import Any, Dict, Iterable, Optional, Tuple, cast

import pandas as pd
import pandera as pa
from astropy.time import Time
from pandera.typing import DataFrame

from .filter import select_cutout
from .io import download_cutout, find_cutouts
from .io.types import CutoutRequest, CutoutRequestSchema
from .plot import plot_cutouts

logger = logging.getLogger("cutouts")


OBSCODE_TOLERANCE_MAPPING = {
    "I41": 1e-8,
    "Q55": 1e-5,
    "W84": 1e-8,
    "V00": 1e-8,
    "695": 1e-8,
}


@pa.check_types
def get_cutouts(
    cutout_requests: DataFrame[CutoutRequestSchema],
    out_dir: str = "~/.cutouts",
    timeout: Optional[int] = 180,
    full_image_timeout: Optional[int] = 600,
    use_cache: bool = True,
    download_full_image: bool = False,
) -> Iterable[Dict[str, Any]]:
    """ """

    # Get urls and metadata for each cutout
    logger.info(f"Getting cutouts for {len(cutout_requests)} requests.")

    results = []
    for record in cutout_requests.to_dict(orient="records"):
        try:
            cutout_request = CutoutRequest(**record)  # type: ignore
            results_df = find_cutouts(cutout_request)
            result = select_cutout(results_df, cutout_request)

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

        for path in result["cutout_image_path"]:
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
    
        if download_full_image == True:
            for path in result["full_image_path"]:
                if path.exists():
                    if use_cache:
                        logger.info(
                            f"{path} already exists locally and using cache, skipping"
                        )
                        continue

                try:
                    download_cutout(
                        result["image_url"],
                        out_file=path.as_posix(),
                        cache=True,
                        pkgname="cutouts",
                        timeout=full_image_timeout,
                    )
                except FileNotFoundError as e:
                    result["error"] = str(e)

    return results


def generate_local_image_paths(result: dict) -> Tuple[str, str]:
    exposure_id_str = ""
    if result["exposure_id"] is not None:
        exposure_id_str = f"_expid_{result['exposure_id']}"

    exposure_mjd_isot = Time(
        result["exposure_start_mjd"], format="mjd", scale="utc"
    ).utc.isot
    cutout_path = f"cutout_{exposure_mjd_isot}_ra{result['ra_deg']:.8f}_dec{result['dec_deg']:.8f}{exposure_id_str}_h{result['height_arcsec']}_w{result['width_arcsec']}.fits"  # noqa: E501
    full_image_path = f"{exposure_mjd_isot}_ra{result['ra_deg']:.8f}_dec{result['dec_deg']:.8f}{exposure_id_str}_h{result['height_arcsec']}_w{result['width_arcsec']}.fits"  # noqa: E501

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
    cutout_height_arcsec: float = 20.0,
    cutout_width_arcsec: float = 20.0,
    download_full_image: bool = False,
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
        cutout_requests["height_arcsec"] = cutout_height_arcsec
    if "width_arcsec" not in cutout_requests:
        cutout_requests["width_arcsec"] = cutout_width_arcsec
    cutout_requests["height_arcsec"].fillna(cutout_height_arcsec, inplace=True)
    cutout_requests["width_arcsec"].fillna(cutout_width_arcsec, inplace=True)

    if "delta_time" not in cutout_requests:
        cutout_requests["delta_time"] = cutout_requests["observatory_code"].apply(
            lambda x: OBSCODE_TOLERANCE_MAPPING[x]
        )
    cutout_requests["delta_time"].fillna(1e-8, inplace=True)

    cutout_results = get_cutouts(
        cast(DataFrame[CutoutRequestSchema], cutout_requests),
        out_dir=str(out_dir), download_full_image=download_full_image,
    )

    plot_candidates = []
    for i, result in enumerate(cutout_results):
        if "error" in result:
            candidate = {
                "path": None,
                "ra": observations["pred_ra_deg"].values[i],
                "dec": observations["pred_dec_deg"].values[i],
                "vra": observations["pred_vra_degpday"].values[i],
                "vdec": observations["pred_vdec_degpday"].values[i],
                "mag": observations["mag"].values[i],
                "mag_sigma": observations["mag_sigma"].values[i],
                "filter": observations["filter"].values[i],
                "exposure_start": observations["exposure_mjd_start"].values[i],
                "exposure_duration": observations["exposure_duration"].values[i],
                "exposure_id": observations["exposure_id"].values[i],
            }
        else:
            candidate = {
                "path": result["cutout_image_path"],
                "ra": observations["pred_ra_deg"].values[i],
                "dec": observations["pred_dec_deg"].values[i],
                "vra": observations["pred_vra_degpday"].values[i],
                "vdec": observations["pred_vdec_degpday"].values[i],
                "mag": observations["mag"].values[i],
                "mag_sigma": observations["mag_sigma"].values[i],
                "filter": observations["filter"].values[i],
                "exposure_start": result["exposure_start_mjd"],
                "exposure_duration": result["exposure_duration"],
            }
        plot_candidates.append(candidate)

    plot_candidates = pd.DataFrame(plot_candidates)
    # Plot cutouts
    fig, ax = plot_cutouts(
        plot_candidates,
        cutout_height_arcsec=cutout_height_arcsec,
        cutout_width_arcsec=cutout_width_arcsec,
    )
    fig.savefig(pathlib.Path(out_dir).joinpath(pathlib.Path(out_file)), bbox_inches="tight")

    return cutout_results
