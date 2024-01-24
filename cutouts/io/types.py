from typing import Optional

from pydantic import BaseModel


class CutoutRequest(BaseModel):
    """
    A single cutout request
    """

    request_id: Optional[str]
    observatory_code: str
    exposure_start_mjd: float
    ra_deg: float
    dec_deg: float
    filter: Optional[str]
    exposure_id: Optional[str]
    exposure_duration: Optional[float]
    height_arcsec: float
    width_arcsec: float
    delta_time: float


class CutoutResult(BaseModel):
    """
    Reference to a single cutout result
    """

    cutout_url: str
    dec_deg: Optional[float]
    exposure_duration: float
    exposure_id: Optional[str]
    exposure_start_mjd: float
    filter: Optional[str]
    height_arcsec: float
    image_url: str
    ra_deg: Optional[float]
    request_id: Optional[str]
    width_arcsec: float
