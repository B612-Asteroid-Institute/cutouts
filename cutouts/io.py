import os
import shutil
import logging
import numpy as np
import numpy.typing as npt
from typing import (
    List,
    Optional
)
from astropy.utils.data import download_file

logger = logging.getLogger(__file__)

def exposure_id_from_url(url):
    id_start = url.find("siaRef=")
    id_end = url.find("&extn")
    exposure_id = url[id_start + 6 : id_end - 1].split(".")[0]
    return exposure_id

def construct_url(
        service,
        dataset,
        exposure_id,
        ra,
        dec,
        height,
        width,
        preview
    ):
    url = f"{service}?col={dataset}&siaRef={exposure_id}.fits.fz&extn=7&POS={ra:.10f},{dec:.10f}&SIZE={height/3600},{width/3600}&preview={preview}"
    return url

def find_cutout(
        ra,
        dec,
        mjd_utc,
        sia_service,
        delta_time=1e-8,
        height=30,
        width=30,
        exposure_id=None
    ):
    center = (ra, dec)
    result = sia_service.search(center, size=(height/3600., width/3600.)).to_table()
    if "mjd_obs" in result.columns:
        mjd_utcs = np.array(result["mjd_obs"]).astype(float)
    else:
        mjd_utcs = np.array(result["mjd_utc_obs"]).astype(float)

    row = result[(mjd_utcs <= mjd_utc + delta_time) & (mjd_utcs >= mjd_utc - delta_time) & (result["prodtype"] == "image")]
    if len(row) == 0:
        err = ("No cutout found.")
        raise FileNotFoundError(err)

    cutout_url = row["access_url"].value[0]

    if exposure_id is not None:
        url_exposure_id = exposure_id_from_url(cutout_url)

        if exposure_id != url_exposure_id:
            logger.warning(
                f"Exposure ID ({url_exposure_id}) found via search on RA, Dec," \
                f"and MJD_utc does not match the given exposure ID ({exposure_id})."
            )

    return cutout_url

def download_cutout(
        url: str,
        out_file: str = None,
        show_progress: bool = True,
        timeout: Optional[int] = None,
        http_headers=None,
        ssl_context=None,
        allow_insecure=False
    ) -> str:

    path = download_file(
        url,
        cache=True,
        pkgname="cutouts",
        show_progress=show_progress,
        timeout=timeout,
        http_headers=http_headers,
        ssl_context=ssl_context,
        allow_insecure=allow_insecure

    )
    if out_file is not None:
        os.makedirs(
            os.path.dirname(out_file),
            exist_ok=True
        )
        shutil.copy(path, out_file)
        path = out_file

    return path
