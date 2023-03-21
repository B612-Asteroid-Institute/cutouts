import os
from contextlib import contextmanager
from unittest.mock import patch

from ..types import CutoutRequest
from ..ztf import find_cutouts_ztf

# 2022 YP2 (2021-09-11T08:50:29.000)
# Result file : testdata/ztf/ztf_77.4640676_29.8956146_59940.320462999865.csv
cutout_request1 = CutoutRequest(
    request_id="2022YP2",
    observatory_code="I41",
    exposure_start_mjd=59940.320462999865,
    ra_deg=77.4640676,
    dec_deg=29.8956146,
    filter="r",
    exposure_id="ztf_exp_59940.32064",
    exposure_duration=30.0,
    height_arcsec=20.0,
    width_arcsec=20.0,
    delta_time=1e-8,
)
TEST_REQUESTS = [cutout_request1]


@contextmanager
def mock_ztf_query(csv_file: str):
    with patch("cutouts.io.ztf.perform_request") as mock_query:
        csv_file = os.path.join(os.path.dirname(__file__), "testdata", "ztf", csv_file)
        with open(csv_file, "r") as f:
            mock_query.return_value = f.read()
        yield mock_query


def test_ztf_query():
    with mock_ztf_query("ztf_77.4640676_29.8956146_59940.320462999865.csv") as mock:
        results = find_cutouts_ztf(cutout_request1)
        mock.assert_called_once()

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
