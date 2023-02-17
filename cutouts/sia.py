import logging

import pandas as pd
from pyvo.dal import SIAResults
from pyvo.dal.sia import SIAService

logger = logging.getLogger(__file__)


class SIAHandler:
    def __init__(
        self,
        sia_url: str,
    ):
        """
        Class to handle Simple Image Access (SIA) queries.

        This allows for a common interface to a SIA service, when differing SIA services
        do not conform to the same response format.

        Parameters
        ----------
        sia_url : str
            Simple Image Access (SIA) service URL.

        """
        self.sia_url = sia_url
        self.sia_service = SIAService(sia_url)
        return

    def search(
            self,
            ra_deg: float,
            dec_deg: float,
            height_arcsec: float = 20,
            width_arcsec: float = 20
            ) -> pd.DataFrame:
        """
        Search for cutouts for a given RA, Dec.

        The SIA service returns a table of cutouts. This method returns a 
        pandas DataFrame of the table.

        Parameters
        ----------
        ra_deg : float
            Right Ascension in degrees.
        dec_deg : float
            Declination in degrees.
        height_arcsec : float, optional
            Height of cutout in arcseconds.
        width_arcsec : float, optional
            Width of cutout in arcseconds.

        """
        result = self.sia_service.search(
            (ra_deg, dec_deg),
            size=(height_arcsec/3600., width_arcsec/3600.)
        )
        if isinstance(result, SIAResults):
            result = pd.DataFrame(result)
        else:
            result = result.to_table().to_pandas()
        # Filter out non-image results
        if "prodtype" in result.columns:
            result = result[result["prodtype"] == "image"]

        # Normalize access_url column
        if "access_url" not in result.columns:
            if "get_fits" in result.columns:
                result["access_url"] = result["get_fits"]
        return result
