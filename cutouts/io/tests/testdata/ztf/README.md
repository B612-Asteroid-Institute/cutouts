# ZTF Cutouts Service

To generate these query results:

```python
from cutouts.io.ztf import perform_request
from cutouts.io.types import CutoutRequest
from cutouts.io.tests.test_ztf import TEST_REQUESTS

ZTF_URL_BASE = "https://irsa.ipac.caltech.edu/ibe/search/ztf/products/sci"

for cutout_request in TEST_REQUESTS:
    out_csv = f"ztf_{cutout_request.ra_deg}_{cutout_request.dec_deg}_{cutout_request.exposure_start_mjd}.csv"

    width_deg = cutout_request.width_arcsec / 3600
    height_deg = cutout_request.height_arcsec / 3600
    search_url = f"{ZTF_URL_BASE}?POS={cutout_request.ra_deg},{cutout_request.dec_deg}&SIZE={width_deg},{height_deg}&ct=csv"  # noqa: E501

    content = perform_request(search_url)
    with open(out_csv, "w") as f:
        f.write(content)
```
