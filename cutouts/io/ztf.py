import io
import logging
from typing import Tuple

import backoff
import pandas as pd
import pandera as pa
import requests
from pandera.typing import DataFrame

from .types import CutoutRequest, CutoutsResultSchema

logger = logging.getLogger(__file__)


def generate_ztf_image_urls_for_result(
    ra_deg: float,
    dec_deg: float,
    height_arcsec: float,
    width_arcsec: float,
    filefracday: float,
    field: int,
    ccdid: int,
    imgtypecode: str,
    filtercode: str,
    qid: int,
) -> Tuple[str, str]:
    """
    Generate a cutout URL from a ZTF SIA result.

    Parameters
    ----------
    result : `~pandas.DataFrame`
        Dataframe with SIA query results.

    Returns
    -------
    cutout_url : str
        URL to cutout
    """
    filefracday_str = str(filefracday)
    year = str(filefracday_str)[:4]
    monthday = str(filefracday_str)[4:8]
    fracday = str(filefracday_str)[8:]
    paddedfield = str(field).zfill(6)
    paddedccdid = str(ccdid).zfill(2)
    imgtypecode = str(imgtypecode)
    filtercode = str(filtercode)
    qid_str = str(qid)

    image_url = (
        f"https://irsa.ipac.caltech.edu/ibe/data/ztf/products/sci/"
        f"{year}/"
        f"{monthday}/"
        f"{fracday}/"
        f"ztf_{filefracday_str}_"
        f"{paddedfield}_"
        f"{filtercode}_"
        f"c{paddedccdid}_"
        f"{imgtypecode}_"
        f"q{qid_str}_"
        f"sciimg.fits"
    )

    cutout_url = f"{image_url}?center={ra_deg},{dec_deg}&size={height_arcsec},{width_arcsec}arcsec&gzip=false"
    return image_url, cutout_url


@backoff.on_exception(
    backoff.expo,
    (
        requests.exceptions.Timeout,
        requests.exceptions.ConnectionError,
        requests.exceptions.HTTPError,
    ),
    max_tries=5,
)
def perform_request(search_url):
    logger.info(f"Performing request to {search_url}")
    response = requests.get(search_url)
    response.raise_for_status()
    return response.text


@pa.check_types()
def find_cutouts_ztf(cutout_request: CutoutRequest) -> DataFrame[CutoutsResultSchema]:
    """
    Search the ZTF service for cutouts and images at a given RA, Dec.

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
        f"Fetching ZTF cutout with ra: {cutout_request.ra_deg} dec: {cutout_request.dec_deg}."
    )

    ZTF_URL_BASE = "https://irsa.ipac.caltech.edu/ibe/search/ztf/products/sci"

    width_deg = cutout_request.width_arcsec / 3600
    height_deg = cutout_request.height_arcsec / 3600

    search_url = f"{ZTF_URL_BASE}?POS={cutout_request.ra_deg},{cutout_request.dec_deg}&SIZE={width_deg},{height_deg}&ct=csv"  # noqa: E501

    content = perform_request(search_url)
    results = pd.read_csv(io.StringIO(content))
    results["exposure_start_mjd"] = results["obsjd"].values.astype(float) - 2400000.5
    results.rename(
        columns={
            "filtercode": "filter",
            "ra": "ra_deg",
            "dec": "dec_deg",
            "expid": "exposure_id",
            "exptime": "exposure_duration",
        },
        inplace=True,
    )

    # Cast the exposure_id to a string
    results["exposure_id"] = results["exposure_id"].astype(str)

    logger.info(f"ZTF query returned table with {len(results)} row.")
    if len(results) == 0:
        err = "No cutouts found."
        raise FileNotFoundError(err)

    # Assign values based on the image metadata to form the url
    full_image_urls = []
    cutout_urls = []

    for result in results.to_dict(orient="records"):
        full_image_url, cutout_url = generate_ztf_image_urls_for_result(
            ra_deg=cutout_request.ra_deg,
            dec_deg=cutout_request.dec_deg,
            height_arcsec=cutout_request.height_arcsec,
            width_arcsec=cutout_request.width_arcsec,
            filefracday=result["filefracday"],
            field=result["field"],
            ccdid=result["ccdid"],
            imgtypecode=result["imgtypecode"],
            filtercode=result["filter"],
            qid=result["qid"],
        )
        full_image_urls.append(full_image_url)
        cutout_urls.append(cutout_url)

    results["cutout_url"] = cutout_urls
    results["image_url"] = full_image_urls

    # Populate the height and results from the request not
    # the results from the query
    results["height_arcsec"] = cutout_request.height_arcsec
    results["width_arcsec"] = cutout_request.width_arcsec

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
