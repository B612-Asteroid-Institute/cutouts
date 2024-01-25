from typing import Optional

from pydantic import BaseModel


class CutoutRequest(BaseModel):
    """
    A single cutout request
    """

    request_id: Optional[str] = None
    observatory_code: str
    exposure_start_mjd: float
    ra_deg: float
    dec_deg: float
    filter: Optional[str] = None
    exposure_id: Optional[str] = None
    exposure_duration: Optional[float] = None
    height_arcsec: float
    width_arcsec: float
    delta_time: float


class CutoutResult(BaseModel):
    """
    Reference to a single cutout result
    """

    cutout_url: str
    dec_deg: Optional[float] = None
    exposure_duration: float
    exposure_id: Optional[str] = None
    exposure_start_mjd: float
    filter: Optional[str] = None
    height_arcsec: float
    image_url: str
    ra_deg: Optional[float] = None
    request_id: Optional[str] = None
    width_arcsec: float
