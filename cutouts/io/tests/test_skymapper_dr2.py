import os
from contextlib import contextmanager
from unittest.mock import patch

from astropy.io.votable import parse
from pyvo.dal.sia import SIAResults

from ..skymapper import SKYMAPPER_DR2_SIA, find_cutouts_skymapper_dr2
from ..types import CutoutRequest

# 2021 EZ3 (2015-09-11T08:50:29.000)
# Result file : testdata/skymapper_dr2/skymapper_dr2_282.4358586_-16.1105754_57276.3683912.xml
cutout_request1 = CutoutRequest(
    request_id=None,
    observatory_code="Q55",
    exposure_start_mjd=57276.3683912,
    ra_deg=282.4358586,
    dec_deg=-16.1105754,
    filter="z",
    exposure_id="20150911085027",
    exposure_duration=100.0,
    height_arcsec=20.0,
    width_arcsec=20.0,
    delta_time=1e-05,
)

TEST_REQUESTS = [cutout_request1]


@contextmanager
def mock_sia_skymapper_dr2_query(table_file: str):
    with patch.object(SKYMAPPER_DR2_SIA, "search") as mock_query:
        table_file = os.path.join(
            os.path.dirname(__file__), "testdata", "skymapper_dr2", table_file
        )
        response_table = parse(table_file)
        mock_query.return_value = SIAResults(response_table)
        yield mock_query


def test_sia_skymapper_dr2_query():

    with mock_sia_skymapper_dr2_query(
        "skymapper_dr2_282.4358586_-16.1105754_57276.3683912.xml"
    ) as mock:
        results = find_cutouts_skymapper_dr2(cutout_request1)
        mock.assert_called_once_with(
            cutout_request1.ra_deg,
            cutout_request1.dec_deg,
            cutout_request1.height_arcsec,
            cutout_request1.width_arcsec,
        )

        for col in [
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
        ]:
            assert col in results.columns
