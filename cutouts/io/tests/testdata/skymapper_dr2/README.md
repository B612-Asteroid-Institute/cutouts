# SkyMapper DR2 SIA Service

To generate these query results:

```python
from pyvo.dal.sia import SIAService
from cutouts.io.types import CutoutRequest
from cutouts.io.tests.test_skymapper_dr2 import TEST_REQUESTS

SIA_URL = "https://api.skymapper.nci.org.au/public/siap/dr2/query?"
service = SIAService(SIA_URL)

for cutout_request in TEST_REQUESTS:
    out_xml = f"skymapper_dr2_{cutout_request.ra_deg}_{cutout_request.dec_deg}_{cutout_request.exposure_start_mjd}.xml"
    results = service.search(
        (cutout_request.ra_deg, cutout_request.dec_deg),
        size=(cutout_request.height_arcsec / 3600.0, cutout_request.width_arcsec / 3600.0),
        format="all",
        center="overlaps"
    )
    results._votable.to_xml(out_xml)
```
