from ..io import exposure_id_from_url

def test_exposure_id_from_url():
    ### Test several URLs to make sure the exposure ID is correctly read.
    exposure_id = "c4d_130905_013936_ooi_g_ls9"
    url = "http://datalab.noirlab.edu/svc/cutout?col=nsc_dr2&siaRef=c4d_130905_013936_ooi_g_ls9.fits.fz&extn=43&POS=331.5320874009333,0.564165914713085&SIZE=0.005555555555555556,0.005555555555555556"
    assert exposure_id_from_url(url) == exposure_id

    exposure_id = "c4d_130911_051211_ooi_i_d2"
    url = "http://datalab.noirlab.edu/svc/cutout?col=nsc_dr2&siaRef=c4d_130911_051211_ooi_i_d2.fits.fz&extn=13&POS=330.2129586953439,0.45337483161655867"
    assert exposure_id_from_url(url) == exposure_id

    exposure_id = "c4d_130930_005720_ooi_i_d2"
    url = "http://datalab.noirlab.edu/svc/cutout?col=nsc_dr2&siaRef=c4d_130930_005720_ooi_i_d2.fits.fz"
    assert exposure_id_from_url(url) == exposure_id