from typing import Optional

import pandera as pa
from pandera.typing import Series
from pydantic import BaseModel


class CutoutRequestSchema(pa.SchemaModel):
    """
    Dataframe validation for multiple cutout requests
    """

    request_id: Optional[Series[str]] = pa.Field(nullable=True)
    observatory_code: Series[str] = pa.Field(coerce=True)
    exposure_start_mjd: Series[float] = pa.Field(nullable=False, coerce=True)
    ra_deg: Series[float] = pa.Field(ge=0, le=360, coerce=True)
    dec_deg: Series[float] = pa.Field(ge=-90, le=90, coerce=True)
    filter: Optional[
        Series[str]
    ] = pa.Field()  # TODO: validate against real list of filters?
    exposure_id: Optional[Series[str]] = pa.Field(nullable=True)
    exposure_duration: Optional[Series[float]] = pa.Field(
        ge=0, le=2000, coerce=True, nullable=True
    )
    height_arcsec: Series[float] = pa.Field(ge=0, le=200, coerce=True, nullable=True)
    width_arcsec: Series[float] = pa.Field(ge=0, le=200, coerce=True, nullable=True)
    delta_time: Series[float] = pa.Field(ge=0, le=100, coerce=True, nullable=True)


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
