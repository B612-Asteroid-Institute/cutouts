import os
from contextlib import contextmanager
from unittest.mock import patch
import pathlib
import pickle
import pandas as pd

from astropy.io.votable import parse
from pyvo.dal.sia import SIAResults

from ..nsc import NSC_DR2_SIA, find_cutouts_nsc_dr2
from ...main import get_cutouts
from ..types import CutoutRequest

# 2014 HE199 (2014-04-28T08:07:52.435)
# Contains weird VR filter results
# Result file : testdata/nsc_dr2/nsc_dr2_227.5251615214173_-27.026013823449265_56775.33880132809.xml
cutout_request1 = CutoutRequest(
    request_id=None,
    observatory_code="W84",
    exposure_start_mjd=56775.33880132809,
    ra_deg=227.5251615214173,
    dec_deg=-27.026013823449265,
    filter="VR",
    exposure_id="c4d_140428_080931_ooi_VR_v1",
    exposure_duration=40.0,
    height_arcsec=20.0,
    width_arcsec=20.0,
    delta_time=1e-08,
)

TEST_REQUESTS = [cutout_request1]


@contextmanager
def mock_sia_nsc_dr2_query(table_file: str):
    with patch.object(NSC_DR2_SIA, "search") as mock_query:
        table_file = os.path.join(
            os.path.dirname(__file__), "testdata", "nsc_dr2", table_file
        )
        response_table = parse(table_file)
        mock_query.return_value = SIAResults(response_table)
        yield mock_query


def test_sia_nsc_dr2_query():

    with mock_sia_nsc_dr2_query(
        "nsc_dr2_227.5251615214173_-27.026013823449265_56775.33880132809.xml"
    ) as mock:
        results = find_cutouts_nsc_dr2(cutout_request1)
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

        assert "VR" in results["filter"].values


def test_sia_nsc_dr2_query_():

    with mock_sia_nsc_dr2_query(
        "nsc_dr2_227.5251615214173_-27.026013823449265_56775.33880132809.xml"
    ) as mock:
        results, comparison_results = get_cutouts(
            pd.DataFrame(cutout_request1),
            out_dir=pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
        )
        print(results)
        print(comparison_results)

        with open('cutout_results.pickle', 'wb') as f:
            pickle.dump(results, f)
        with open('cutout_comparison_results.pickle', 'wb') as f:
            pickle.dump(comparison_results, f)

        assert False