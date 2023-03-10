import logging
from abc import ABCMeta, abstractmethod

from pyvo.dal.sia import SIAResults, SIAService

logger = logging.getLogger(__file__)


class SIAHandler(metaclass=ABCMeta):
    def __init__(
        self,
    ):
        """ """
        self.sia_service = SIAService(self.SIA_URL)

    SIA_URL = ""

    @abstractmethod
    def search(
        self,
        ra_deg: float,
        dec_deg: float,
        height_arcsec: float = 20,
        width_arcsec: float = 20,
    ) -> SIAResults:
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
        pass
