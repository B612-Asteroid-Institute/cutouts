from cutouts.io.types import CutoutRequest
from cutouts.io.ztf import find_cutouts_ztf


def test_find_cutouts_ztf():
    cutout_request = CutoutRequest(
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
    return find_cutouts_ztf(cutout_request)
